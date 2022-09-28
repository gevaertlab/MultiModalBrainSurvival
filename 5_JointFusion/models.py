from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

class BagHistopathologyRNAModel(nn.Module):
    def __init__(self, resnet, rna_mlp, final_mlp):
        super(BagHistopathologyRNAModel, self).__init__()
        self.resnet = resnet
        self.rna_mlp = rna_mlp
        self.final_mlp = final_mlp

    def forward(self, patch_bag, rna,resnet_dim=2048):
        (batch_size, bag_size, c, h, w) = patch_bag.shape
        patch_bag = patch_bag.reshape(-1, c, h, w)
        image_features = self.resnet.forward_extract(patch_bag)
        image_features = image_features.view(batch_size, bag_size, resnet_dim)
        image_features = image_features.mean(dim=1) 
        rna_features = self.rna_mlp(rna)
        #print (image_features.shape)
        #print(rna_features.shape)
        output = self.final_mlp(torch.cat([image_features, rna_features], dim=1))
        return output

class HistopathologyRNAModel(nn.Module):
    def __init__(self, resnet, rna_mlp, final_mlp):
        super(HistopathologyRNAModel, self).__init__()
        self.resnet = resnet
        self.rna_mlp = rna_mlp
        self.final_mlp = final_mlp

    def forward(self, patch, rna):
        image_features = self.resnet.forward_extract(patch)
        rna_features = self.rna_mlp(rna)
        output = self.final_mlp(torch.cat([image_features, rna_features], dim=1))
        return output    

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