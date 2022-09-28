########################################################
### MERGE HISTOPATH AND GENE EXPRESSION UNIMODAL SCORES
#########################################################
### This script merges the survival scores of the histopathology and gene expression unimodal models
### --> this is the input of the Late Fusion model
### - input: 
###          - FFPE model scores (from 3_HistoPath_savescore.py)
###          - RNA model scores (from 2_GeneExpress_savescore.py)
### - output:  Concatenated FFPE and RNA model scores for Late Fusion model
###############################################################################
###############################################################################
### Example command
### $ python 1_MergeScores.py
###################################################
###################################################

### Set Environment
####################
import pandas as pd
import numpy as np

### Merge Scores
#################
## Pathology
path_df = pd.read_csv("savescore/ffpe_scores.csv", header=0)
path_df.rename({'score':'path_score', 'id':'case'}, inplace=True, axis=1)
    
## RNA
rna_df = pd.read_csv("savescore/rna_scores.csv", header=0)
rna_df.rename({'score':'rna_score', 'id':'case'}, inplace=True, axis=1)

### Merge
final_df = path_df.merge(rna_df[['case','rna_score']], how="inner", on="case")
final_df.drop(['Unnamed: 0'], axis=1).to_csv("combined_scores.csv", index=False)