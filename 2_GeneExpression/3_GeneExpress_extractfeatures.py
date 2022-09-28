#####################################################
### MODEL FEATURES FROM GENE EXPRESSION MODULE (RNA)
####################################################
### This script extracts the model features of the rna model 
### - input: log+z-score transformed RNA values of 12,778 genes (see genes.txt) + config file
### - output: model features per sample
###############################################################################
###############################################################################
### Example command
### $ 3_GeneExpress_extractfeatures.py --config "/path/to/config_rna_extractfeatures.json"
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

def extract_features(model, val_dataloader, device):

    ## Validation
    model.eval()

    features_list = []
    wsi_list = []
    case_list = []

    for b_idx, batch_dict in enumerate(val_dataloader):
        inputs = batch_dict['rna_data'].to(device)
        input_size = inputs.size()
        with torch.no_grad():
            features = model.extract(inputs)            
        features_list.append(features.detach().cpu().numpy())
        case_list.append(batch_dict['case'])

    case_list = [c for c_b in case_list for c in c_b]
    features_list = np.concatenate(features_list, axis=0)

    case_uniques = []
    features_final = []
    for case in set(case_list):
        case_uniques.append(case)
        l = np.array([(x == case) for x in case_list])
        feature = l.T.dot(features_list)/(l.sum())
        features_final.append(feature)

    features_final = np.asarray(features_final)
    return case_uniques,features_final


def main():

    # parse args and load config
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config['use_cuda']) else "cpu")

    model_rna = torch.nn.Sequential(
        nn.Dropout(), 
        nn.Linear(12778, 4096), 
        nn.ReLU(), 
        nn.Dropout(), 
        nn.Linear(4096, 2048) 
    )

    combine_mlp = torch.nn.Sequential(nn.Linear(2048, 1))
    model = RNAOnlyModel(model_rna, combine_mlp)
    
    if config['model_path'] != "":
        model.load_state_dict(torch.load(config['model_path']))
        print("Loaded model from checkpoint")


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
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x],num_workers=config["num_workers"])
        for x in['train', 'val', 'test']}

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU
    model = model.to(device)
    for dataset in ["val", "train", "test"]:
        print("extracting features for dataset : {}".format(dataset))
        cases,features = extract_features(model,dataloaders_dict[dataset],device)

        #outname = config['flag'].split("_")[-1]
        outname = config['flag']
        if 'cv' in outname:
            pd.DataFrame(cases).to_csv(os.path.join(config['output_path'],"rna_cases_{}_{}.csv".format(dataset,outname)))
            np.savetxt(os.path.join(config['output_path'],"rna_features_{}_{}.csv".format(dataset,outname)).format(dataset),
                features,delimiter=",")
        else:
            pd.DataFrame(cases).to_csv(os.path.join(config['output_path'],"rna_cases_{}.csv".format(dataset)))
            np.savetxt(os.path.join(config['output_path'],"rna_features_{}.csv".format(dataset)).format(dataset),
                features,delimiter=",")

### Input arguments
####################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='example_config.json', help='configuration json file')
parser.add_argument("--seed",type=int,default=1111)

### MAIN
##########

if __name__ == '__main__':
    main()
