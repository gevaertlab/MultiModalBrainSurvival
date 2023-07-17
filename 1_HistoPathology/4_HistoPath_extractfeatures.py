#####################################################
### MODEL FEATURES FROM HISTOPATHOLOGY MODULE (FFPE)
####################################################
### This script extracts the model features of the histopathology model 
### - input: 224x224 patches (see 1_WSI2Patches.py) + config file
### - output: model features per sample
###############################################################################
###############################################################################
### Example command
### $ python 4_HistoPath_extractfeatures.py --config "/path/to/config_ffpe_extractfeatures.json"
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
from models import AggregationModel, Identity, CoxLoss, PatchBagDataset

from tensorboardX import SummaryWriter

plt.switch_backend('agg')

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def extract_features(model, val_dataloader, device):
    ## Validation

    model.eval()

    features_list = []
    wsi_list = []
    case_list = []

    for b_idx, batch_dict in enumerate(val_dataloader):
        inputs = batch_dict['patch_bag'].to(device)
        # for binary classification, target_label = 'label', for multiclass target_label = 'label_multiclass'
        wsis = batch_dict['WSI']
        input_size = inputs.size()

        # forward

        with torch.no_grad():
            features, attention_weights = model.extract(inputs)
            
        features_list.append(features.detach().cpu().numpy())
        wsi_list.append(wsis)
        case_list.append(batch_dict['case'])

    wsi_list = [w for w_b in wsi_list for w in w_b]
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
    # torch.multiprocessing.set_sharing_strategy('file_system')

    with open(args.config) as f:
        config = json.load(f)
    if 'flag' in config:
        args.flag = config['flag']
    if args.flag == "":
        args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config['use_cuda']) else "cpu")

    resnet = resnet50(pretrained=config['pretrained'])
    aggregator = None
    if config['aggregator'] == 'identity':
        aggregator = Identity()
    elif config['aggregator'] == "attention":
        aggregator = TanhAttention(dim=2048)
    elif config['aggregator'] == 'transformer':
        aggregator = TransformerEncoder(config['transformer_layers'], 2048, config['aggregator_hdim'], 5,
                                        config['aggregator_hdim'], .2, 0)
    model = AggregationModel(resnet=resnet, aggregator=aggregator, aggregator_dim=config['aggregator_hdim'],
                             resnet_dim=2048, out_features=config['num_classes'])
    if config['model_path'] != "":
        model.load_state_dict(torch.load(config['model_path']))
        print("Loaded model from checkpoint")
    else:
        print("Loaded model pretrained on imagenet")
    print(type(model))

    input_size = 224
    # create Datasets and DataLoaders
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
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
        image_datasets['train'] = PatchBagDataset(patch_data_path=config["data_path"], csv_path=config["train_csv_path"],
                                            img_size=config["img_size"], bag_size=config['val_bag_size'],
                                            transforms=data_transforms['val'],
                                            max_patches_total=config.get('max_patch_per_wsi_val', 1000))
        image_samplers['train'] = SequentialSampler(image_datasets['train'])
    
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
        list(image_datasets.keys())}

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU
    model = model.to(device)
    for dataset in list(image_datasets.keys()):
        print("extracting features for dataset : {}".format(dataset))
        cases,features = extract_features(model,dataloaders_dict[dataset],device)

        #outname = config['flag'].split("_")[-1]
        outname = config['flag']
        if 'cv' in config['flag']:
            pd.DataFrame(cases).to_csv(os.path.join(config['output_path'],"pathology_cases_{}_{}.csv".format(dataset,outname)))
            np.savetxt(os.path.join(config['output_path'],"pathology_features_{}_{}.csv".format(dataset,outname)).format(dataset),
                features,delimiter=",")
        else:
            pd.DataFrame(cases).to_csv(os.path.join(config['output_path'],
                                                "pathology_cases_{}.csv".format(dataset)))
            np.savetxt(os.path.join(config['output_path'],
                                "pathology_features_{}.csv".format(dataset)).format(dataset),
                features,delimiter=",")
        

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='example_config.json', help='configuration json file')
parser.add_argument("--seed",type=int,default=1111)


if __name__ == '__main__':
    main()
