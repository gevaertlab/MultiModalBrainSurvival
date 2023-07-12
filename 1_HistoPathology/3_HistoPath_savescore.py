#################################################
### MODEL SCORES FOR HISTOPATHOLOGY MODEL (FFPE)
#################################################
### This script fetches the model scores of the histopathology model 
### - input: 224x224 patches (see 1_WSI2Patches.py) + config file
### - output: model survival predictions per sample
###############################################################################
###############################################################################
### Example command
### $ 3_HistoPath_savescore.py --config "/path/to/config_ffpe_savescore.json"
###################################################
###################################################

### Set Environment
####################

from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler
import torch.multiprocessing
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import *
from scipy.special import softmax as scipy_softmax
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

from resnet import resnet50
from models import AggregationModel, Identity, TanhAttention, CoxLoss, PatchBagDataset, NLLSurvLoss

from tensorboardX import SummaryWriter

plt.switch_backend('agg')

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

### Functions
###############

def evaluate(model, val_dataloader, device, criterion, num_classes, mode='val', task='survival_bin'):
    
    ## Validation
    model.eval()

    output_list = []
    wsi_list = []
    case_list = []
    loss_list = []
    labels_list = []
    survival_months_list = []
    survival_bin_list = []
    vital_status_list = []

    for b_idx, batch_dict in enumerate(val_dataloader):

        #print(b_idx)

        inputs = batch_dict['patch_bag'].to(device)
        wsis = batch_dict['WSI']
        survival_months = batch_dict['survival_months'].to(device).float()
        survival_bin = batch_dict['survival_bin'].to(device).long()
        vital_status = batch_dict['vital_status'].to(device).float()
        input_size = inputs.size()

        # forward
        with torch.no_grad():
            outputs, attention_weights = model.forward(inputs) #output for patch

            if task == 'survival_bin':
                censoring = 1 - vital_status
                loss = criterion(h=outputs, y=survival_bin, c=censoring)
            elif task == 'survival_prediction':
                loss = criterion(outputs.view(-1), survival_months, vital_status)

        loss_list.append(loss.item())
        output_list.append(outputs.detach().cpu().numpy())
        #print(len(output_list))
        survival_months_list.append(survival_months.detach().cpu().numpy())
        survival_bin_list.append(survival_bin.detach().cpu().numpy())
        vital_status_list.append(vital_status.detach().cpu().numpy())
        wsi_list.append(wsis)
        #print(len(wsi_list))
        case_list.append(batch_dict['case'])

    # link patch scores to wsi / case ids
    wsi_list = [w for w_b in wsi_list for w in w_b]
    case_list = [c for c_b in case_list for c in c_b]
    survival_months_list = np.array([s for s_b in survival_months_list for s in s_b])
    survival_bin_list = np.array([b for s_b in survival_bin_list for b in s_b])
    vital_status_list = np.array([v for v_b in vital_status_list for v in v_b])

    output_list = np.concatenate(output_list, axis=0)

    if task == 'survival_bin':

        wsi_CI, _ = get_nllsurv_CI(output_list, vital_status_list, survival_months_list, wsi_list, num_classes)
        case_CI, pandas_output = get_nllsurv_CI(output_list, vital_status_list, survival_months_list, case_list, num_classes)
        #print("{} wsi  | CI {:.3f}".format(mode, wsi_CI))

    elif task == 'survival_prediction':
        
        wsi_CI, _ = get_survival_CI(output_list, wsi_list, survival_months_list, vital_status_list)
        case_CI, pandas_output = get_survival_CI(output_list, case_list, survival_months_list, vital_status_list)
        #print("{} wsi | CI {:.3f}".format(mode, wsi_CI))
        #print("{} case| CI {:.3f}".format(mode, case_CI))

    val_loss = np.mean(loss_list)

    return val_loss, wsi_CI, case_CI, pandas_output
    #return val_loss, wsi_CI

def get_survival_CI(output_list, ids_list, survival_months, vital_status):
    
    ids_unique = sorted(list(set(ids_list)))
    id_to_scores = {}
    id_to_scores_mean = {}
    id_to_survival_months = {}
    id_to_vital_status = {}

    for i in range(len(output_list)):
        id = ids_list[i]
        id_to_scores[id] = id_to_scores.get(id, []) + [output_list[i, 0]]
        id_to_survival_months[id] = survival_months[i]
        id_to_vital_status[id] = vital_status[i]

    for k in ids_unique:
        id_to_scores_mean[k] = np.mean(id_to_scores[k])

    score_list = np.array([id_to_scores_mean[id] for id in ids_unique])
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

    for i in range(len(ids_unique)):
        k = ids_unique[i]
        for j in range(0, num_classes):
            #print(id_to_scores[k][j])
            score_tensor[i][j] = torch.from_numpy(np.asarray(id_to_scores[k][j])).to(score_tensor)

    survival_months_list = np.array(survival_months_list)
    vital_status_list = np.array(vital_status_list)
    #print(len(survival_months_list)) #242
    #print(len(vital_status_list)) #242
    
    # Predict CI
    hazards = torch.sigmoid(score_tensor)
    survival = torch.cumprod(1 - hazards, dim=-1)
    risk_all = -torch.sum(survival, dim=-1).detach().cpu().numpy().flatten()
    
    conc_index = concordance_index_censored(
        vital_status_list.astype(bool), survival_months_list, risk_all, tied_tol=1e-08)[0]
    
    pandas_output = pd.DataFrame({'id': ids_unique, 'score': risk_all, 'survival_months': survival_months_list,
                                  'vital_status': vital_status_list})
    #print(pandas_output)
    #for j in range(0, num_classes):
    #    #print(score_list['score_{}'.format(j)])
    #    pandas_output['score_{}'.format(j)] = np.array(score_list['score_{}'.format(j)])

    return conc_index, pandas_output

def main():
    # parse args and load config
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config['use_cuda']) else "cpu")
    
    print("Loading model")

    resnet = resnet50(pretrained=config['pretrained'])
    aggregator = None
    if config['aggregator'] == 'identity':
        aggregator = Identity()
    elif config['aggregator'] == "attention":
        aggregator = TanhAttention(dim=2048)
    elif config['aggregator'] == 'transformer':
        aggregator = TransformerEncoder(config['transformer_layers'], 2048, config['aggregator_hdim'], 5,
                                        config['aggregator_hdim'], .2, 0)
    model = AggregationModel(resnet=resnet, aggregator=aggregator, aggregator_dim=config['aggregator_hdim'],resnet_dim=2048, out_features=config['num_classes'])

    if 'model_path' in config:
        model.load_state_dict(torch.load(config['model_path']))
        print("Loaded model from checkpoint")
    else:
        config['model_path'] = ""

    if 'output_path' not in config:
        config['output_path'] = ""

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
    
    print("Loading datasets")

    if 'train_csv_path' in config:
        image_datasets['train'] = PatchBagDataset(patch_data_path=config["data_path"], csv_path=config["train_csv_path"],
                                              img_size=config["img_size"],
                                              bag_size=config['train_bag_size'],
                                              transforms=data_transforms['train'],
                                              max_patches_total=config.get('max_patch_per_wsi_train', 1000))
        image_samplers['train'] = RandomSampler(image_datasets['train'])
        
    if 'val_csv_path' in config:
        image_datasets['val'] = PatchBagDataset(patch_data_path=config["data_path"], csv_path=config["val_csv_path"],
                                             img_size=config["img_size"], bag_size=config['val_bag_size'],
                                             transforms=data_transforms['val'],
                                             max_patches_total=config.get('max_patch_per_wsi_val', 1000))
        image_samplers['val'] = SequentialSampler(image_datasets['val'])


    if 'test_csv_path' in config:
        image_datasets['test'] = PatchBagDataset(patch_data_path=config["data_path"], csv_path=config["test_csv_path"],
                                             img_size=config["img_size"], bag_size=config['val_bag_size'],
                                             transforms=data_transforms['val'],
                                             max_patches_total=config.get('max_patch_per_wsi_val', 1000))
        image_samplers['test'] = SequentialSampler(image_datasets['test'])

    print("loaded datasets")
    

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x],
                                       num_workers=config["num_workers"])
        for x in
        list(image_datasets.keys())
        #['train', 'val', 'test']
        #['test']
    }

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU
    model = model.to(device)

     # Setup the loss fxn
    if config['task'] == 'survival_prediction':
        criterion = CoxLoss()
    elif config['task'] == 'survival_bin':
        criterion = NLLSurvLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for dataset in list(image_datasets.keys()):
    #for dataset in ['test']:
        print("Evaluation for dataset : {}".format(dataset))
        dataset_loss, wsi_CI, case_CI, output = evaluate(model,dataloaders_dict[dataset], device, criterion, config['num_classes'], mode='val', task=config['task'])
        print('loss: {}'.format(dataset_loss))
        print('WSI CI: {}'.format(wsi_CI))
        print('case CI: {}'.format(case_CI))
        print('output_path: ' + config['output_path'])
        print('model: ' + config['model_path'].split("/")[-1])

        #outname = config['flag'].split("_")[-1]
        outname = config['flag']
        if 'cv' in outname:
            output.to_csv(config['output_path']+config['model_path'].split("/")[-1]+"_pathology_"+dataset+"_"+outname+"_df.csv")
        else:
            output.to_csv(config['output_path']+config['model_path'].split("/")[-1]+"_pathology_"+dataset+"_df.csv")

### Input arguments
####################   

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,default="",help="configuration file")
parser.add_argument("--seed",type=int,default=1111)

### MAIN
##########

if __name__ == '__main__':
    main()

