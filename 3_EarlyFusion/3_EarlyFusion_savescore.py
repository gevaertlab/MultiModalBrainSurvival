#################################################
### MODEL SCORES FOR EARLY FUSION MODEL (FFPE+RNA)
##################################################
### This script fetches the model scores of the multimodal early fusion model 
### - input: Concatenated FFPE and RNA features (from 1_Concat2Features.py) + config file
### - output:  model survival predictions per sample
###############################################################################
###############################################################################
### Example command
### $ python 3_EarlyFusion_savescore.py --config "/path/to/config_early_savescore.json"
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
from datasets import featureDataset

from tensorboardX import SummaryWriter

plt.switch_backend('agg')

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

### Functions
###############

def evaluate(model, val_dataloader, device, num_classes, mode='val', task='survival_bin'):

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

        inputs = batch_dict['feature_data'].to(device)

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

    case_list = [c for c_b in case_list for c in c_b]
    wsi_list=case_list
    survival_months_list = np.array([s for s_b in survival_months_list for s in s_b])
    survival_bin_list = np.array([b for s_b in survival_bin_list for b in s_b])
    vital_status_list = np.array([v for v_b in vital_status_list for v in v_b])

    output_list = np.concatenate(output_list, axis=0)
            
    if task == 'survival_prediction':
            
        case_CI, pandas_output = get_survival_CI(output_list, case_list, survival_months_list, vital_status_list)
        print("{} case  |  CI {:.3f}".format(mode, case_CI))

    elif task == 'survival_bin':

        case_CI, pandas_output = get_nllsurv_CI(output_list, vital_status_list, survival_months_list, case_list, num_classes)
        print("{} case  |  CI {:.3f}".format(mode, case_CI)) 

    val_loss = np.mean(loss_list)

    return val_loss, case_CI, pandas_output
            
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

def main():
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    #device = torch.device("cpu")
    num_classes = config['num_classes']

    model_feature = torch.nn.Sequential(
        nn.Dropout(), 
        nn.Linear(4096, 2048), 
        nn.ReLU(), 
        nn.Dropout(), 
        nn.Linear(2048, 200), 
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(200, num_classes),
    )

    model = model_feature
    
    if config['model_path'] != "":
        model.load_state_dict(torch.load(config['model_path']))
        print("Loaded model from checkpoint")

    # Create training and validation datasets
    image_datasets = {}
    image_samplers = {}

    if 'train_csv_path' in config:
        image_datasets['train'] = featureDataset(config["train_csv_path"])
        image_samplers['train'] = RandomSampler(image_datasets['train'])

    if 'val_csv_path' in config:
        image_datasets['val'] = featureDataset(config["val_csv_path"])
        image_samplers['val'] = SequentialSampler(image_datasets['val'])

    if 'test_csv_path' in config:    
        image_datasets['test'] = featureDataset(config["test_csv_path"])
        image_samplers['test'] = SequentialSampler(image_datasets['test'])
    
    print("loaded datasets")
    
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
                            image_datasets[x], 
                            batch_size=config['batch_size'], 
                            sampler=image_samplers[x],
                            num_workers=config["num_workers"]
                            )
                        for x in list(image_datasets.keys())
                        }

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU/CPU
    model = model.to(device)

    for dataset in list(image_datasets.keys()):

        print("Evaluation for dataset : {}".format(dataset))

        dataset_loss, case_CI, output = evaluate(model,dataloaders_dict[dataset], device, num_classes, mode='val', task=config['task'])
        print('loss: {}'.format(dataset_loss))
        print('case CI: {}'.format(case_CI))
        print('output_path: ' + config['output_path'])
        print('model: ' + config['model_path'].split("/")[-1])

        #outname = config['flag'].split("_")[-1]
        outname = config['flag']
        if 'cv' in outname:
            output.to_csv(config['output_path']+config['model_path'].split("/")[-1]+"_feature_"+dataset+"_"+outname+"_df.csv")
        else:
            output.to_csv(config['output_path']+config['model_path'].split("/")[-1]+"_feature_"+dataset+"_df.csv")

### Input arguments
####################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,default="",help="configuration file")
parser.add_argument("--seed",type=int,default=4242)

### MAIN
##########

if __name__ == '__main__':
    main()
