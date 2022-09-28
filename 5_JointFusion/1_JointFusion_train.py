#########################################
### TRAIN JOINT FUSION MODEL (FFPE+RNA)
#########################################
### This script trains the multimodal joint fusion model 
### - input: 
###         - 224x224 patches (see 1_WSI2Patches.py)
###         - log+z-score transformed RNA values of 12,778 genes (see genes.txt) + config file
###         - config file
### - output:  Joint Fusion model for survival predictions
###############################################################################
###############################################################################
### Example command
### $ 1_JointFusion_train.py --config "/path/to/config_joint_train.json"
###################################################
###################################################

### Set Environment
####################
from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torch.optim import Adam
import torch.multiprocessing
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import *
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lifelines.utils import concordance_index
from scipy.special import softmax as scipy_softmax

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import os
import copy
import json
import argparse

from datasets import PatchRNADataset, PatchBagRNADataset
from resnet import resnet50
from models import HistopathologyRNAModel, AggregationModel, AggregationProjectModel, Identity, TanhAttention, CoxLoss, BagHistopathologyRNAModel

from tensorboardX import SummaryWriter

plt.switch_backend('agg')

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

### Functions
###############

def evaluate(model, val_dataloader, criterion, summary_writer, device, epoch, mode='val', task='classification', target_label='label'):
    
    ## Validation
    model.eval()

    output_list = []
    wsi_list = []
    case_list = []
    loss_list = []
    labels_list = []
    survival_months_list = []
    vital_status_list = []

    for b_idx, batch_dict in enumerate(val_dataloader):
        patches = batch_dict['patch_bag'].to(device)
        rna_data = batch_dict['rna_data'].to(device)
        survival_months = batch_dict['survival_months'].to(device).float()
        vital_status = batch_dict['vital_status'].to(device).float()
        input_size = patches.size()
        wsis = batch_dict['wsi_file_name']
        labels = batch_dict[target_label].to(device).long()

        # forward
        with torch.no_grad():
            outputs = model(patches, rna_data)
            if task == 'classification':
                loss = criterion(outputs, labels)
            else:
                assert task == 'survival_prediction'
                loss = criterion(outputs.view(-1), survival_months, vital_status)

        loss_list.append(loss.item())
        output_list.append(outputs.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
        survival_months_list.append(survival_months.detach().cpu().numpy())
        vital_status_list.append(vital_status.detach().cpu().numpy())
        wsi_list.append(wsis)
        case_list.append(batch_dict['case'])

    wsi_list = [w for w_b in wsi_list for w in w_b]
    case_list = [c for c_b in case_list for c in c_b]
    labels_list = np.array([l for l_b in labels_list for l in l_b])
    survival_months_list = np.array([s for s_b in survival_months_list for s in s_b])
    vital_status_list = np.array([v for v_b in vital_status_list for v in v_b])

    output_list = np.concatenate(output_list, axis=0)

    if task == 'classification':
        if output_list.shape[1] == 2:
            patch_acc, patch_f1, patch_auc = accuracy_score(labels_list, output_list[:, 1] > .5), f1_score(labels_list,
                                                                                                           output_list[
                                                                                                           :,
                                                                                                           1] > .5), roc_auc_score(
                labels_list, output_list[:, 1])
            print("{} patch  | epoch {} | acc {:.3f} | f1 {:.3f} | auc {:.3f}".format(mode, epoch, patch_acc, patch_f1,
                                                                                      patch_auc))
        wsi_acc, wsi_f1, wsi_auc, pandas_output = get_classification_scores(output_list, wsi_list,
                                                                            labels_list)
        case_acc, case_f1, case_auc, _ = get_classification_scores(output_list, case_list, labels_list)

        print("{} wsi  | epoch {} | acc {:.3f} | f1 {:.3f} | auc {:.3f}".format(mode, epoch, wsi_acc, wsi_f1, wsi_auc))
        print("{} case  | epoch {} | acc {:.3f} | f1 {:.3f} | auc {:.3f}".format(mode, epoch, case_acc, case_f1,
                                                                                 case_auc))

    elif task == 'survival_prediction':
        wsi_CI, pandas_output = get_survival_CI(output_list, wsi_list, survival_months_list, vital_status_list)
        case_CI, _ = get_survival_CI(output_list, case_list, survival_months_list, vital_status_list)
        print("{} wsi  | epoch {} | CI {:.3f}".format(mode, epoch, wsi_CI))
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


def train_model(model, dataloaders, criterion, optimizer, device, save_dir='checkpoints/models/',
                num_epochs=25, summary_writer=None, log_interval=100, task='classification', output_dir=None,target_label='label'):
    best_val_loss = np.inf
    summary_step = 0
    best_epoch = -1

    delta_loss = -1
    counter = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        ## TRAIN
        model.train()

        running_loss = 0.0
        running_corrects = 0.0
        inputs_seen = 0.0
        total_seen = 0.0

        # for logging
        last_running_loss = 0.0
        last_running_corrects = 0.0
        last_time = time.time()

        # Iterate over data.
        for b_idx, batch_dict in enumerate(dataloaders['train']):
            patches = batch_dict['patch_bag'].to(device)
            rna_data = batch_dict['rna_data'].to(device)
            survival_months = batch_dict['survival_months'].to(device).float()
            vital_status = batch_dict['vital_status'].to(device).float()
            input_size = patches.size()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(patches, rna_data)
            if task == 'classification':
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += (preds == labels).sum().item()

            elif task == 'survival_prediction':
                loss = criterion(outputs.view(-1), survival_months, vital_status)
            # backward + optimize only if in training phase

            loss.backward()
            optimizer.step()
            summary_step += 1

            # statistics
            running_loss += loss.item() * input_size[0]

            inputs_seen += input_size[0]
            total_seen += input_size[0]

            if (summary_step % log_interval == 0):
                loss_to_log = (running_loss - last_running_loss) / (inputs_seen)

                acc_to_log = (running_corrects - last_running_corrects) / (inputs_seen)
                speed_to_log = log_interval * input_size[0] / (time.time() - last_time)

                last_time = time.time()
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", loss_to_log, summary_step)
                    summary_writer.add_scalar("train/acc", acc_to_log, summary_step)

                last_running_loss = running_loss
                last_running_corrects = running_corrects
                inputs_seen = 0.0
                print(
                    "train | epoch {0} | batch {4}/{5}| loss {1:10.3f} | acc {2:10.3f} |{3:10.3f} bags /s".format(
                        epoch, loss_to_log,
                        acc_to_log,
                        speed_to_log, b_idx, len(dataloaders['train'])))

        epoch_loss = running_loss / total_seen
        epoch_acc = running_corrects / total_seen

        print('TRAIN Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        train_loss, _ = evaluate(model, dataloaders['train'], criterion, summary_writer, device, epoch,mode='train', task=task,target_label=target_label)
        val_loss, _ = evaluate(model, dataloaders['val'], criterion, summary_writer, device, epoch, mode='val', task=task,target_label=target_label)
        
        if val_loss < best_val_loss:
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
            best_val_loss = val_loss
                

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_last.pt'))

    print("EVALUATING ON VAL SET")
    val_loss, val_output_last = evaluate(model, dataloaders['val'], criterion, summary_writer, device, epoch, task=task,
                                     mode='val', target_label=target_label)

    print("\n")
    print("LOADING BEST MODEL, best epoch = {}".format(best_epoch))
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))

    print("EVALUATING ON VAL SET")
    test_loss, val_output_best = evaluate(model, dataloaders['val'], criterion, summary_writer, device, best_epoch,
                                          task=task,
                                          mode='val', target_label=target_label)

    print("EVALUATING ON TEST SET".format(best_epoch))
    test_loss, test_output_best = evaluate(model, dataloaders['test'], criterion, summary_writer, device, best_epoch,
                                           task=task, mode='test', target_label=target_label)

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        val_output_last.to_csv(os.path.join(output_dir, 'val_output_last.csv'), index=False)

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

    with open(args.config) as f:
        config = json.load(f)
    if 'flag' in config:
        args.flag = config['flag']
    if args.flag == "":
        args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
    if 'checkpoint_path' in config:
        args.checkpoint_path = config['checkpoint_path']
    if 'summary_path' in config:
        args.summary_path = config['summary_path']

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config['use_cuda']) else "cpu")
    num_classes, num_epochs = config['num_classes'], config['num_epochs']

    resnet = resnet50(pretrained=config['pretrained'])
    aggregator = None
    if config['aggregator'] == 'identity':
        aggregator = Identity()
    elif config['aggregator'] == "attention":
        aggregator = TanhAttention(dim=2048)
    elif config['aggregator'] == 'transformer':
        aggregator = TransformerEncoder(config['transformer_layers'], 2048, config['aggregator_hdim'], 5,
                                        config['aggregator_hdim'], .2, 0)
        
    model_histo = resnet
    model_rna = torch.nn.Sequential(
        nn.Dropout(), 
        nn.Linear(12778, 4096), 
        nn.ReLU(), 
        nn.Dropout(), 
        nn.Linear(4096, 2048), 
    )
    combine_mlp = torch.nn.Sequential(
        nn.Dropout(0.8), 
        nn.Linear(4096, 1)) #third model

    model = BagHistopathologyRNAModel(model_histo, model_rna, combine_mlp)
    if config['restore_path'] != "":
        restore_path=config['restore_path']
        print (restore_path)
        model.load_state_dict(torch.load(restore_path))
        print("Loaded model from checkpoint for finetuning")
    
    print("Loaded model")

    input_size = 224
    # create Datasets and DataLoaders
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {}
    image_samplers = {}
    print("loading datasets")

    if args.quick:
        config['max_patch_per_wsi_train'] = 20
        config['max_patch_per_wsi_val'] = 20

    image_datasets['train'] = PatchBagRNADataset(patch_data_path=config["data_path"], csv_path=config["train_csv_path"],img_size=config["img_size"],transforms=data_transforms['train'], bag_size=config['train_bag_size'],max_patch_per_wsi=config.get('max_patch_per_wsi_train', 1000))

    image_datasets['val'] = PatchBagRNADataset(patch_data_path=config["data_path"], csv_path=config["val_csv_path"],img_size=config["img_size"], bag_size=config['val_bag_size'], transforms=data_transforms['val'],max_patch_per_wsi=config.get('max_patch_per_wsi_val', 1000))

    image_datasets['test'] = PatchBagRNADataset(patch_data_path=config["data_path"], csv_path=config["test_csv_path"],img_size=config["img_size"],bag_size=config['val_bag_size'], transforms=data_transforms['val'],max_patch_per_wsi=config.get('max_patch_per_wsi_val', 1000))

    print("loaded datasets")
    image_samplers['train'] = RandomSampler(image_datasets['train'])
    image_samplers['val'] = SequentialSampler(image_datasets['val'])
    image_samplers['test'] = SequentialSampler(image_datasets['test'])

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x],num_workers=config["num_workers"]) for x in ['train', 'val', 'test']}

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU
    model = model.to(device)  

    params_to_update_histo = []
    params_to_update_rna = []
    params_to_update_mlp = []

    print("params to learn")

    n_layers_to_train = config.get('n_layers_to_train', 100)

    layers_to_train = [resnet.fc, resnet.layer4, resnet.layer3, resnet.layer2, resnet.layer1, resnet.conv1]
    layers_to_train = layers_to_train[:n_layers_to_train] 

    for param in model_histo.parameters():
        param.requires_grad = False
    for layer in layers_to_train:
        for n, param in layer.named_parameters():
            param.requires_grad = True
    print("model_histo:")
    for layer in layers_to_train:
        for n, param in layer.named_parameters():
            if param.requires_grad:
                print("\t {}".format(n))
                params_to_update_histo.append(param)
    print("model_rna:")
    for n, param in model_rna.named_parameters():
        if param.requires_grad:
            print("\t {}".format(n))
            params_to_update_rna.append(param)
    print("model integrated:")
    for n, param in combine_mlp.named_parameters():
        if param.requires_grad:
            print("\t {}".format(n))
            params_to_update_mlp.append(param)

    optimizer_ft = Adam([{'params': params_to_update_histo, 'lr': config['lr_histo']},
                        {'params': params_to_update_rna, 'lr': config['lr_rna']},
                        {'params': params_to_update_mlp, 'lr': config['lr_mlp']}],
                        weight_decay=config['weight_decay'])

    # Setup the loss fxn
    if config['task'] == 'survival_prediction':
        criterion = CoxLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    if args.log:
        summary_writer = SummaryWriter(
            os.path.join(args.summary_path,
                         datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_{0}".format(args.flag)))

        summary_writer.add_text("config", str(config))
    else:
        summary_writer = None

    if not os.path.isdir(os.path.join(args.checkpoint_path, 'models', args.flag)):
        os.makedirs(os.path.join(args.checkpoint_path, 'models', args.flag))

    #train_loss, _ = evaluate(model, dataloaders_dict['train'], criterion, summary_writer, device, -1,mode='train', task=config.get('task', 'classification'),target_label=config.get('target_label', 'vital_status'))
    #val_loss, _ = evaluate(model, dataloaders_dict['val'], criterion, summary_writer, device, -1, mode='val', task=config.get('task', 'classification'),target_label=config.get('target_label', 'vital_status'))

    train_model(model=model, dataloaders=dataloaders_dict, criterion=criterion,
                optimizer=optimizer_ft,
                device=device,
                num_epochs=num_epochs, summary_writer=summary_writer,
                save_dir=os.path.join(args.checkpoint_path, 'models', args.flag),
                task=config.get('task', 'classification'),
                output_dir=os.path.join(args.checkpoint_path, 'outputs', args.flag),
                target_label=config.get('target_label', 'vital_status')
                
               )

### Input arguments
####################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')
parser.add_argument("--save_images", type=int, default=0, help='save sample images from the dataset')
parser.add_argument("--quick", type=int, default=0, help='use small datasets to check that the script runs')
parser.add_argument("--log", type=int, default=0, help='0 = do not use a summary writer')
parser.add_argument("--seed", type=int, default=1111, help="seed for the random number generator")

### MAIN
##########

if __name__ == '__main__':
    main()
