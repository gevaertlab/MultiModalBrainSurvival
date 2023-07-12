from torchvision import models, transforms
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
from sksurv.metrics import concordance_index_censored
import random

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        out = x
        attention_weights = torch.ones(x.shape[0], x.shape[1], device=x.device)
        return out, attention_weights

class TanhAttention(nn.Module):
    def __init__(self, dim=2048):
        super(TanhAttention, self).__init__()
        self.dim = dim
        self.vector = torch.nn.Parameter(torch.zeros(dim))
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        logits = torch.tanh(self.linear(x)).matmul(self.vector.unsqueeze(-1))
        attention_weights = torch.nn.functional.softmax(logits, dim=1)
        out = x * attention_weights * x.shape[1]
        return out,attention_weights

class AggregationModel(nn.Module):
    def __init__(self, resnet, aggregator, aggregator_dim, resnet_dim=2048, out_features=1):
        super(AggregationModel, self).__init__()
        self.resnet = resnet
        self.aggregator = aggregator
        self.fc = nn.Linear(aggregator_dim, out_features)
        self.aggregator_dim = aggregator_dim
        self.resnet_dim = resnet_dim

    def forward(self, x):
        features,attention_weights  = self.extract(x)
        out = self.fc(features)
        return out, attention_weights

    def extract(self,x):
        (batch_size, bag_size, c, h, w) = x.shape
        x = x.reshape(-1, c, h, w)
        features = self.resnet.forward_extract(x)
        features = features.view(batch_size, bag_size, self.resnet_dim)  # bsize, bagsize, resnet_dim

        features, attention_weights = self.aggregator(features)  # bsize, bagsize, aggregator_dim
        features = features.mean(dim=1)  # batch_size,self.hdim
        return features,attention_weights

class AggregationProjectModel(nn.Module):
    def __init__(self, resnet, aggregator, aggregator_dim, resnet_dim=2048, out_features=1,hdim=200,dropout=.3):
        super(AggregationProjectModel, self).__init__()
        self.resnet = resnet
        self.aggregator = aggregator        
        self.aggregator_dim = aggregator_dim
        self.resnet_dim = resnet_dim
        self.hdim = hdim
        self.dropout = nn.Dropout(p=dropout)
        self.project = nn.Linear(aggregator_dim, hdim)
        self.fc = nn.Linear(hdim, out_features)

    def forward(self, x):
        features,attention_weights  = self.extract(x)
        out = self.fc(features)
        return out, attention_weights

    def extract(self,x):
        (batch_size, bag_size, c, h, w) = x.shape
        x = x.reshape(-1, c, h, w)
        features = self.resnet.forward_extract(x)
        features = features.view(batch_size, bag_size, self.resnet_dim)  # bsize, bagsize, resnet_dim

        features, attention_weights = self.aggregator(features)  # bsize, bagsize, aggregator_dim
        features = features.mean(dim=1)  # batch_size,aggregator_dim
        features = self.project(features)
        features = F.tanh(features)
        features = self.dropout(features)
           
        return features,attention_weights

def cox_loss(cox_scores, times, status):
    '''

    :param cox_scores: cox scores, size (batch_size)
    :param times: event times (either death or censor), size batch_size
    :param status: event status (1 for death, 0 for censor), size batch_size
    :return: loss of size 1, the sum of cox losses for the batch
    '''

    times, sorted_indices = torch.sort(-times)
    cox_scores = cox_scores[sorted_indices]
    status = status[sorted_indices]
    cox_scores = cox_scores -torch.max(cox_scores)
    exp_scores = torch.exp(cox_scores)
    loss = cox_scores - torch.log(torch.cumsum(exp_scores, dim=0)+1e-5)
    loss = - loss * status

    if (loss != loss).any():
        import pdb;
        pdb.set_trace()

    return loss.mean()

class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss,self).__init__()

    def forward(self,cox_scores,times,status):
        return cox_loss(cox_scores,times,status)

# adapted from https://github.com/mahmoodlab/PORPOISE/blob/master/utils/loss_func.py
class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        Set importance of censored loss
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """

    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y: (n_batches, 1)
            The true time bin index label.
        c: (n_batches, 1)
            The censoring status indicator.
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)

def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        Set importance of censored loss
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    #print("h shape", h.shape)

    # make sure these are ints
    #y = y.type(torch.int64)
    #c = c.type(torch.int64)

    batch_size = len(y)
    y = y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1

    hazards = torch.sigmoid(h)

    S = torch.cumprod(1 - hazards, dim=1) #surival is cumulative product of 1 - hazards

    #S = S.type(torch.int64)
    #print(S.dtype)
    #print(c.dtype)
    #print(1.dtype)

    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]

    #https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    #s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    #h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    #s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    # c = 0 is uncensored (deceased)
    #uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    # c = 1 is censored (living)
    #censored_loss = - c * torch.log(s_this)

    # c = 0 is uncensored (deceased)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, y).clamp(min=eps)))
    # c = 1 is censored (living)
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, y+1).clamp(min=eps))

    # neg_l = censored_loss + uncensored_loss
    # if alpha is not None:
    #     loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = (1 - alpha)*censored_loss + uncensored_loss

    #print(loss.mean())
    #print(loss.sum())

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss

class PatchBagDataset(Dataset):
    """
    csv_path must contain csv with header
    case, wsi_file_name, attr1,...,attrk


    """

    def __init__(self, patch_data_path, csv_path, img_size, transforms=None, bag_size=40, max_patches_total=1000):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.bag_size = bag_size
        self.max_patches_total = max_patches_total
        self.data = {}
        self.index = []
        self.preprocess()

    def preprocess(self):
        A = pd.read_csv(self.csv_path)
        for i, row in A.iterrows():
            row = row.to_dict()
            WSI = row['wsi_file_name'].split('.')[0]#+'.'+row['wsi_file_name'].split('.')[1]
            n_patches = sum(1 for _ in open(os.path.join(self.patch_data_path, WSI, 'loc.txt'))) - 2
            n_patches = min(n_patches,self.max_patches_total)
            images = [os.path.join(self.patch_data_path, WSI, WSI + "_patch_{}.png".format(i)) for i in range(n_patches)]
            #images = [os.path.join(self.patch_data_path, WSI, WSI + "_patch_{}.png".format(i)) for i in random.sample(range(sum(1 for _ in open(os.path.join(self.patch_data_path, WSI, 'loc.txt'))) - 2),n_patches)]      

                
            self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, "n_images": len(images)})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, self.bag_size * k))

    def shuffle(self):
        for k in self.data.keys():
            wsi_row = self.data[k]
            np.random.shuffle(wsi_row['images'])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        
        
        (WSI, i) = self.index[idx]
        imgs = []
        row = self.data[WSI]
        for patch in row['images'][i:i + self.bag_size]:
            with open(patch, "rb") as f:
                img = Image.open(f).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            imgs.append(img)
       
        img = torch.stack(imgs, dim=0)
        result = row.copy()
        result['patch_bag'] = img
        
               
        return result 
