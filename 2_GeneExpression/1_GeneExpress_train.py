######################################
### TRAIN GENE EXPRESSION MODEL (RNA)
######################################
### This script trains the unimodal gene expression model 
### - input: log+z-score transformed RNA values of 12,778 genes (see genes.txt) + config file
### - output:  RNA MLP module for survival predictions
###############################################################################
###############################################################################
### Example command
### $ python 1_GeneExpress_train.py --config "/path/to/config_rna_train.json"
###################################################
###################################################

### Set Environment
####################

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import Adam, rmsprop
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler
import torch.multiprocessing
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import *
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import os
import copy
import json
import argparse

from models import cox_loss, RNAOnlyModel, nll_loss
from datasets import RNADataset

from tensorboardX import SummaryWriter

plt.switch_backend('agg')

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

### Functions
###############

def evaluate(model, val_dataloader, device, epoch, num_classes, mode='val', task='survival_bin'):

    ## Validation
    model.eval()

    output_list = []
    case_list = []
    loss_list = []
    labels_list = []
    survival_months_list = []
    survival_bin_list = []
    vital_status_list = []

    for b_idx, batch_dict in enumerate(val_dataloader):

        inputs = batch_dict['rna_data'].to(device)

        survival_months = batch_dict['survival_months'].to(device).float()
        survival_bin = batch_dict['survival_bin'].to(device).long()
        vital_status = batch_dict['vital_status'].to(device).float()
        input_size = inputs.size()

        # forward
        with torch.no_grad():

            outputs = model.forward(inputs)

            if task == 'survival_prediction':
                loss = cox_loss(outputs.view(-1), survival_months.view(-1), vital_status.view(-1))

            elif task == 'survival_bin':
                censoring = 1 - vital_status
                loss = nll_loss(h=outputs, y=survival_bin, c=censoring)

        loss_list.append(loss.item())
        output_list.append(outputs.detach().cpu().numpy())
        survival_months_list.append(survival_months.detach().cpu().numpy())
        survival_bin_list.append(survival_bin.detach().cpu().numpy())
        vital_status_list.append(vital_status.detach().cpu().numpy())
        case_list.append(batch_dict['case'])

    case_list = [c for c_b in case_list for c in c_b] #DOUBLE ARRAY
    survival_months_list = np.array([s for s_b in survival_months_list for s in s_b])
    survival_bin_list = np.array([b for s_b in survival_bin_list for b in s_b])
    vital_status_list = np.array([v for v_b in vital_status_list for v in v_b])

    output_list = np.concatenate(output_list, axis=0)
    
    if task == 'survival_prediction':

        case_CI, pandas_output = get_survival_CI(output_list, case_list, survival_months_list, vital_status_list)
        print("{} case  | epoch {} | CI {:.3f}".format(mode, epoch, case_CI))

    elif task == 'survival_bin':

        case_CI, pandas_output = get_nllsurv_CI(output_list, vital_status_list, survival_months_list, case_list, num_classes)
        print("{} case  | epoch {} | CI {:.3f}".format(mode, epoch, case_CI)) 

    val_loss = np.mean(loss_list)

    return val_loss, pandas_output
            
def get_survival_CI(output_list, ids_list, survival_months, vital_status):

    ids_unique = sorted(list(set(ids_list)))
    id_to_scores = {}
    id_to_survival_months = {}
    id_to_vital_status = {}

    for i in range(len(output_list)):
        id = ids_list[i]
        id_to_scores[id] = id_to_scores.get(id, []) + [output_list[i, 0]]
        id_to_survival_months[id] = survival_months[i]
        id_to_vital_status[id] = vital_status[i]

    for k in id_to_scores.keys():
        id_to_scores[k] = np.mean(id_to_scores[k])

    score_list = np.array([id_to_scores[id] for id in ids_unique])
    survival_months_list = np.array([id_to_survival_months[id] for id in ids_unique])
    vital_status_list = np.array([id_to_vital_status[id] for id in ids_unique])

    CI = concordance_index(survival_months_list, -score_list, vital_status_list)
    pandas_output = pd.DataFrame({'id': ids_unique, 'score': score_list, 'survival_months': survival_months_list,
                                  'vital_status': vital_status_list})

    return CI, pandas_output

def get_nllsurv_CI(predictions, vital_status, survival_months, ids_list, num_classes):

    # Process risk scores per wsi / case
    ids_unique = sorted(list(set(ids_list)))
    id_to_scores = {}
    id_to_survival_months = {}
    id_to_vital_status = {}

    for i in range(len(predictions)):
        id = ids_list[i]

        if id not in id_to_scores:
             id_to_scores[id] = {}
             id_to_survival_months[id] = {}
             id_to_vital_status[id] = {}

        for j in range(0, num_classes):

            if j not in id_to_scores[id]:
                id_to_scores[id][j] = []
            id_to_scores[id][j].append(predictions[i, j])
            id_to_survival_months[id][j] = survival_months[i]
            id_to_vital_status[id][j] = vital_status[i]

    score_list = {}
    survival_months_list = []
    vital_status_list = []

    #for k in id_to_scores.keys():
    for k in ids_unique:

        for j in range(0, num_classes):
            
            if 'score_{}'.format(j) not in score_list:
                score_list['score_{}'.format(j)] = []

            id_to_scores[k][j] = np.mean(id_to_scores[k][j])

            score_list['score_{}'.format(j)].append(id_to_scores[k][j])

            if j == 0:
                survival_months_list.append(id_to_survival_months[k][j])
                vital_status_list.append(id_to_vital_status[k][j])
    
    score_tensor = torch.empty((len(ids_unique),num_classes), dtype=torch.float32)
    #print(score_tensor)

    for i in range(len(ids_unique)):
        k = ids_unique[i]
        for j in range(0, num_classes):
            #print(id_to_scores[k][j])
            score_tensor[i][j] = torch.from_numpy(np.asarray(id_to_scores[k][j])).to(score_tensor)

    survival_months_list = np.array(survival_months_list)
    vital_status_list = np.array(vital_status_list)

    # Predict CI
    hazards = torch.sigmoid(score_tensor)
    survival = torch.cumprod(1 - hazards, dim=-1)
    #print(survival)
    #print(survival.shape)
    risk_all = -torch.sum(survival, dim=-1).detach().cpu().numpy().flatten()

    conc_index = concordance_index_censored(
        vital_status_list.astype(bool), survival_months_list, risk_all, tied_tol=1e-08)[0]
    
    pandas_output = pd.DataFrame({'id': ids_unique, 'score': risk_all, 'survival_months': survival_months_list,
                                  'vital_status': vital_status_list})
    #for j in range(0, num_classes):
    #    #print(score_list['score_{}'.format(j)])
    #    pandas_output['score_{}'.format(j)] = np.array(score_list['score_{}'.format(j)])

    return conc_index, pandas_output

def train_model(model, dataloaders, optimizer, device, num_epochs=25, num_classes=4,
                summary_writer=None, log_interval=100, task='survival_bin', 
                save_dir='checkpoints/models', output_dir=None):

    best_val_loss = np.inf
    summary_step = 0
    best_epoch = -1
    #print (len (dataloaders['train']))
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        ## TRAIN
        model.train()
        running_loss = 0.0
        inputs_seen = 0.0
        total_seen = 0.0

        # for logging
        last_running_loss = 0.0
        last_time = time.time()

        # Iterate over data.
        for b_idx, batch in enumerate(dataloaders['train']):
            
            inputs = batch['rna_data'].to(device)
            survival_months = batch['survival_months'].to(device).float()
            survival_bin = batch['survival_bin'].to(device).long()
            vital_status = batch['vital_status'].to(device).float()

            # zero the parameter gradients
            optimizer.zero_grad()
            input_size = inputs.size()
            
            # forward
            outputs = model(inputs)

            if task == 'survival_prediction':
                loss = cox_loss(outputs.view(-1), survival_months.view(-1), vital_status.view(-1))
            elif task == 'survival_bin':
                censoring = 1 - vital_status
                loss = nll_loss(h=outputs, y=survival_bin, c=censoring)

            loss.backward()
            optimizer.step()
            summary_step += 1
            vital_sum = vital_status.sum().item()

            # statistics
            #running_loss += loss.item() * vital_sum
            #inputs_seen += vital_sum
            #total_seen += vital_sum
            running_loss += loss.item() * input_size[0]

            inputs_seen += input_size[0]
            total_seen += input_size[0]

            if (summary_step % log_interval == 0):
                loss_to_log = (running_loss - last_running_loss) / (inputs_seen)
                #speed_to_log = log_interval * input_size[0] / (time.time() - last_time)

                last_time = time.time()
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", loss_to_log, summary_step)

                last_running_loss = running_loss
                inputs_seen = 0.0

                print(
                    "train | epoch {0} | batch {2}/{3}| loss {1:10.3f} ".format(
                        epoch, loss_to_log,
                        b_idx, len(dataloaders['train'])))
                
        epoch_loss = running_loss / total_seen

        print('TRAIN Loss: {:.4f}'.format(epoch_loss))

        train_loss, _ = evaluate(model, dataloaders['train'], device, epoch, num_classes, mode='train', task=task)
        print('TRAIN Loss: {:.4f}'.format(train_loss))
        val_loss, _ = evaluate(model, dataloaders['val'], device, epoch, num_classes, mode='val', task=task)
        print('VAL Loss: {:.4f}'.format(val_loss))

        if val_loss < best_val_loss and epoch > 0:
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
            best_val_loss = val_loss

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_last.pt'))

    print("LAST MODEL, epoch = {}".format(epoch))
    print("EVALUATING ON TRAIN SET")
    test_loss, train_output_last = evaluate(model, dataloaders['train'], device, epoch,
                                          num_classes, task=task,
                                          mode='train')

    print("EVALUATING ON VAL SET")
    test_loss, val_output_last = evaluate(model, dataloaders['val'], device, epoch, 
                                          num_classes, task=task,
                                          mode='val')
    print("EVALUATING ON TEST SET")
    test_loss, test_output_last = evaluate(model, dataloaders['test'], device, epoch,
                                           num_classes, task=task, 
                                           mode='test')

    print("\n")
    print("LOADING BEST MODEL, best epoch = {}".format(best_epoch))
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))

    print("EVALUATING ON TRAIN SET")
    test_loss, train_output_best = evaluate(model, dataloaders['train'], device, best_epoch,
                                          num_classes, task=task,
                                          mode='train')

    print("EVALUATING ON VAL SET")
    test_loss, val_output_best = evaluate(model, dataloaders['val'], device, best_epoch,
                                          num_classes, task=task, 
                                          mode='val')

    print("EVALUATING ON TEST SET")
    test_loss, test_output_best = evaluate(model, dataloaders['test'], device, best_epoch, 
                                           num_classes, task=task,
                                           mode='test')
    
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        train_output_last.to_csv(os.path.join(output_dir, 'train_output_last.csv'), index=False)
        val_output_last.to_csv(os.path.join(output_dir, 'val_output_last.csv'), index=False)
        test_output_last.to_csv(os.path.join(output_dir, 'test_output_last.csv'), index=False)

        train_output_best.to_csv(os.path.join(output_dir, 'train_output_best.csv'), index=False)
        val_output_best.to_csv(os.path.join(output_dir, 'val_output_best.csv'), index=False)
        test_output_best.to_csv(os.path.join(output_dir, 'test_output_best.csv'), index=False)

        print("Wrote model output files to " + output_dir)

    if summary_writer is not None:
        summary_writer.close()

def main():

    # parse args and load config
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    with open(args.config) as f:
        config = json.load(f)
    if 'flag' in config:
        args.flag = config['flag']
    if 'checkpoint_path' in config:
        args.checkpoint_path = config['checkpoint_path']
    if 'summary_path' in config:
        args.summary_path = config['summary_path']
    if args.flag == "":
        args.flag = 'train_coxloss_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config['use_cuda']) else "cpu")
    num_classes, num_epochs = config['num_classes'], config['num_epochs']

    model_rna = torch.nn.Sequential(
        nn.Dropout(), 
        nn.Linear(12778, 4096), 
        nn.ReLU(), 
        nn.Dropout(), 
        nn.Linear(4096, 2048), 

    )

    combine_mlp = torch.nn.Sequential(nn.Linear(2048, num_classes))
    model = RNAOnlyModel(model_rna, combine_mlp)
    
    print("Loaded model")

    # Create training and validation datasets
    image_datasets = {}
    image_samplers = {}
    
    image_datasets['train'] = RNADataset(config["train_csv_path"])
    image_datasets['val'] = RNADataset(config["val_csv_path"])
    image_datasets['test'] = RNADataset(config["test_csv_path"])

    print("loaded datasets")
    image_samplers['train'] = RandomSampler(image_datasets['train'])
    image_samplers['val'] = RandomSampler(image_datasets['val'])
    image_samplers['test'] = RandomSampler(image_datasets['test'])

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x],
                                       num_workers=config["num_workers"])
        for x in
        ['train', 'val', 'test']}

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU
    model = model.to(device)
    if config['restore_path'] != "":
        model.load_state_dict(torch.load(config['restore_path']))
        print("Loaded model from checkpoint for finetuning")

    if config['model_path'] != "":
        model.load_state_dict(torch.load(config['model_path']))
        print("Loaded model from checkpoint for finetuning")

    params_to_update_rna = []
    params_to_update_mlp = []

    print("params to learn")

    for n, param in model_rna.named_parameters():
        if param.requires_grad:
            print("\t {}".format(n))
            params_to_update_rna.append(param)
    for n, param in combine_mlp.named_parameters():
        if param.requires_grad:
            print("\t {}".format(n))
            params_to_update_mlp.append(param)

    optimizer_ft = Adam([{'params': params_to_update_rna, 'lr': config['lr_rna']},
                        {'params': params_to_update_mlp, 'lr': config['lr_mlp']}],
                        weight_decay=config['weight_decay'])

    # Train and evaluate
    if args.log:
        summary_dir = os.path.join(args.summary_path,
                                   datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_{0}".format(args.flag))
        if not os.path.isdir(summary_dir):
            os.makedirs(summary_dir)
        summary_writer = SummaryWriter(summary_dir)
        summary_writer.add_text("config", str(config))
    else:
        summary_writer = None

    if not os.path.isdir(os.path.join(args.checkpoint_path, 'models', args.flag)):
        os.makedirs(os.path.join(args.checkpoint_path, 'models', args.flag))
        
    train_model(model=model, 
                dataloaders=dataloaders_dict,
                optimizer=optimizer_ft,
                device=device,
                num_epochs=num_epochs, 
                num_classes=num_classes,
                summary_writer=summary_writer,
                log_interval=100,
                task=config.get('task', 'survival_bin'), 
                save_dir=os.path.join(args.checkpoint_path, 'models', args.flag),
                output_dir=os.path.join(args.checkpoint_path, 'outputs', args.flag)
                )

### Input arguments
####################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')
parser.add_argument("--save_images", type=int, default=0, help='save sample images from the dataset')
parser.add_argument("--quick", type=int, default=0, help='use small datasets to check that the script runs')
parser.add_argument("--log", type=int, default=0, help='do not use a summary writer')
parser.add_argument("--seed", type=int, default=3333, help="seed for the random number generator")
parser.add_argument("--save_every", type=int, default=-1,
                    help="save mode every k epochs, -1 for saving only the best checkpoints")
parser.add_argument("--no-save", type=int, default=0)

### MAIN
##########

if __name__ == '__main__':
    main()
