################################################
### MODEL SCORES FOR GENE EXPRESSION MODEL (RNA)
################################################
### This script fetches the model scores of the gene expression model 
### - input: log+z-score transformed RNA values of 12,778 genes (see genes.txt) + config file
### - output:  model survival predictions per sample
###############################################################################
###############################################################################
### Example command
### $ 2_GeneExpress_savescore.py --config "/path/to/config_rna_savescore.json"
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import os
import copy
import json
import argparse

from models import cox_loss, RNAOnlyModel
from datasets import RNADataset

from tensorboardX import SummaryWriter

plt.switch_backend('agg')

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

### Functions
###############

def evaluate(model, val_dataloader, device, mode='val'):
    
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
        inputs = batch_dict['rna_data'].to(device)
        survival_months = batch_dict['survival_months'].to(device).float()
        vital_status = batch_dict['vital_status'].to(device).float()
        input_size = inputs.size()

        # forward
        with torch.no_grad():
            outputs = model.forward(inputs)
            loss = cox_loss(outputs.view(-1), survival_months.view(-1), vital_status.view(-1))

        loss_list.append(loss.item())
        output_list.append(outputs.detach().cpu().numpy())
        survival_months_list.append(survival_months.detach().cpu().numpy())
        vital_status_list.append(vital_status.detach().cpu().numpy())
        case_list.append(batch_dict['case'])

    case_list = [c for c_b in case_list for c in c_b]
    wsi_list=case_list
    survival_months_list = np.array([s for s_b in survival_months_list for s in s_b])
    vital_status_list = np.array([v for v_b in vital_status_list for v in v_b])

    output_list = np.concatenate(output_list, axis=0)
            
    #wsi_CI, _ = get_survival_CI(output_list, wsi_list, survival_months_list, vital_status_list)
    case_CI, pandas_output = get_survival_CI(output_list, case_list, survival_months_list, vital_status_list)
    #print("{} wsi  |  CI {:.3f}".format(mode, wsi_CI))
    print("{} case  |  CI {:.3f}".format(mode, case_CI))

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

def main():
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    model_rna = torch.nn.Sequential(
        nn.Dropout(), 
        nn.Linear(12778, 4096), 
        nn.ReLU(), 
        nn.Dropout(), 
        nn.Linear(4096, 2048)
    )

    combine_mlp = torch.nn.Sequential(nn.Linear(2048, 1))
    model = RNAOnlyModel(model_rna, combine_mlp)
    
    print("Loaded model")
    
    if config['model_path'] != "":
        model.load_state_dict(torch.load(config['model_path']))
        print("Loaded model from checkpoint")

    image_datasets = {}
    image_samplers = {}
    batch_size = {}
    
    image_datasets['train'] = RNADataset(config["train_csv_path"])
    image_datasets['val'] = RNADataset(config["val_csv_path"])
    image_datasets['test'] = RNADataset(config["test_csv_path"])

    print("loaded datasets")
    image_samplers['train'] = RandomSampler(image_datasets['train'])
    image_samplers['val'] = RandomSampler(image_datasets['val'])
    image_samplers['test'] = RandomSampler(image_datasets['test'])

    # Create training and validation dataloaders
    batch_size['train']=config['batch_size']
    batch_size['val'] = config ['batch_size']
    batch_size['test'] = config ['batch_size']
    
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size[x], sampler=image_samplers[x],num_workers=1)
        for x in
        ['train', 'val', 'test']}

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU
    model = model.to(device)

    for dataset in ["train", "val", "test"]:

        print("Evaluation for dataset : {}".format(dataset))
        _, output=evaluate (model,dataloaders_dict[dataset],device, mode='val')
        
        #outname = config['flag'].split("_")[-1]
        outname = config['flag']
        if 'cv' in outname:
            output.to_csv(config['output_path']+"rna_"+dataset+"_"+outname+"_df.csv")
        else:
            output.to_csv(config['output_path']+"rna_"+dataset+"_df.csv")

### Input arguments
####################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')
parser.add_argument("--seed",type=int,default=1111)

### MAIN
##########

if __name__ == '__main__':
    main()

    
