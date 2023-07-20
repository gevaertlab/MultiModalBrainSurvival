#########################################
### MODEL SCORES FOR JOINT FUSION MODEL (FFPE+RNA)
#########################################
### This script tfetches the model scores of the multimodal joint fusion model 
### - input: 
###         - 224x224 patches (see 1_WSI2Patches.py)
###         - log+z-score transformed RNA values of 12,778 genes (see genes.txt) + config file
###         - config file
### - output:  model survival predictions per sample
###############################################################################
###############################################################################
### Example command
### $ python 2_JointFusion_savescore.py --config "/path/to/config_joint_savescore.json"
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

from datasets import PatchRNADataset, PatchBagRNADataset
from resnet import resnet50
from models import HistopathologyRNAModel, AggregationModel, AggregationProjectModel, Identity, TanhAttention, CoxLoss, BagHistopathologyRNAModel, NLLSurvLoss

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
    survival_months_list = []
    survival_bin_list = []
    vital_status_list = []

    for b_idx, batch_dict in enumerate(val_dataloader):

        patches = batch_dict['patch_bag'].to(device)
        rna_data = batch_dict['rna_data'].to(device)
        wsis = batch_dict['wsi_file_name']

        survival_months = batch_dict['survival_months'].to(device).float()
        survival_bin = batch_dict['survival_bin'].to(device).long()
        vital_status = batch_dict['vital_status'].to(device).float()
        input_size = patches.size()
        
        with torch.no_grad():

            outputs = model(patches, rna_data)

            if task == 'classification':
                loss = criterion(outputs, labels)

            elif task == 'survival_bin':
                censoring = 1 - vital_status
                loss = criterion(h=outputs, y=survival_bin, c=censoring)

            else:
                assert task == 'survival_prediction'
                loss = criterion(outputs.view(-1), survival_months, vital_status)

        loss_list.append(loss.item())
        output_list.append(outputs.detach().cpu().numpy())
        survival_months_list.append(survival_months.detach().cpu().numpy())
        vital_status_list.append(vital_status.detach().cpu().numpy())
        survival_bin_list.append(survival_bin.detach().cpu().numpy())
        wsi_list.append(wsis)
        case_list.append(batch_dict['case'])

    wsi_list = [w for w_b in wsi_list for w in w_b]
    case_list = [c for c_b in case_list for c in c_b]
    survival_months_list = np.array([s for s_b in survival_months_list for s in s_b])
    survival_bin_list = np.array([b for s_b in survival_bin_list for b in s_b])
    vital_status_list = np.array([v for v_b in vital_status_list for v in v_b])

    output_list = np.concatenate(output_list, axis=0)

    if task == 'survival_prediction':

        wsi_CI, pandas_output = get_survival_CI(output_list, wsi_list, survival_months_list, vital_status_list)
        case_CI, _ = get_survival_CI(output_list, case_list, survival_months_list, vital_status_list)
        
        print("{} wsi  | CI {:.3f}".format(mode, wsi_CI))
        print("{} case | CI {:.3f}".format(mode, case_CI))

    elif task == 'survival_bin':

        wsi_CI, _ = get_nllsurv_CI(output_list, vital_status_list, survival_months_list, wsi_list, num_classes)
        case_CI, pandas_output = get_nllsurv_CI(output_list, vital_status_list, survival_months_list, case_list, num_classes)

        print("{} wsi  | CI {:.3f}".format(mode, wsi_CI))
        print("{} case | CI {:.3f}".format(mode, case_CI))

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
    combine_mlp = torch.nn.Sequential(nn.Dropout(0.8), nn.Linear(4096, num_classes))

    model = BagHistopathologyRNAModel(model_histo, model_rna, combine_mlp)
    if config['model_path'] != "":
        model.load_state_dict(torch.load(config['model_path']))
        print("Loaded model from checkpoint")
    
    print("Loaded model")

    input_size = config['img_size']
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

    if 'train_csv_path' in config:
        image_datasets['train'] = PatchBagRNADataset(patch_data_path=config["data_path"], csv_path=config["train_csv_path"],
                                                img_size=config["img_size"],
                                                transforms=data_transforms['train'], 
                                                bag_size=config['train_bag_size'],
                                                max_patch_per_wsi=config.get('max_patch_per_wsi_train', 1000))
        image_samplers['train'] = RandomSampler(image_datasets['train'])
    
    if 'val_csv_path' in config:
        image_datasets['val'] = PatchBagRNADataset(patch_data_path=config["data_path"], csv_path=config["val_csv_path"],
                                                img_size=config["img_size"], 
                                                bag_size=config['val_bag_size'], 
                                                transforms=data_transforms['val'],
                                                max_patch_per_wsi=config.get('max_patch_per_wsi_val', 1000))
        image_samplers['val'] = SequentialSampler(image_datasets['val'])

    if 'test_csv_path' in config:
        image_datasets['test'] = PatchBagRNADataset(patch_data_path=config["data_path"], csv_path=config["test_csv_path"],
                                                img_size=config["img_size"],
                                                bag_size=config['val_bag_size'], 
                                                transforms=data_transforms['val'],
                                                max_patch_per_wsi=config.get('max_patch_per_wsi_val', 1000))
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
    if config['task'] == 'survival_prediction':
        criterion = CoxLoss()
    elif config['task'] == 'survival_bin':
        criterion = NLLSurvLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for dataset in list(image_datasets.keys()):
        print("Evaluation for dataset : {}".format(dataset))
        _,output = evaluate(model, dataloaders_dict[dataset], device, criterion, num_classes, mode='val', task=config['task'])

        #outname = config['flag'].split("_")[-1]
        outname = config['flag']
        if 'cv' in outname:
            output.to_csv(config['output_path']+config['model_path'].split("/")[-1]+"_joint_"+dataset+"_"+outname+"_df.csv")
        else:
            output.to_csv(config['output_path']+config['model_path'].split("/")[-1]+"_joint_"+dataset+"_df.csv")

### Input arguments
####################

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')
parser.add_argument("--seed",type=int,default=1111)

### MAIN
##########

if __name__ == '__main__':
    main()
