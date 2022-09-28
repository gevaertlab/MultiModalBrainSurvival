from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class RNAOnlyModel(nn.Module):
    def __init__(self, rna_mlp, final_mlp):
        super(RNAOnlyModel, self).__init__()
        self.rna_mlp = rna_mlp
        self.final_mlp = final_mlp

    def forward(self, rna):
        rna_features = self.rna_mlp(rna)
        output = self.final_mlp(rna_features)
        return output
    
    def extract(self,rna):
        rna_features = self.rna_mlp(rna)
        return rna_features
    
    
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
