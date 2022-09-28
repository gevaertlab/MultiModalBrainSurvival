#########################################
### CONCATENATE RNA AND FFPE FEATURES
#########################################
### This script concatenates the FFPE and RNA features which is used as input Early Fusion model 
### - input: 
###          - FFPE featues (from 4_HistoPath_extractfeatures.py)
###          - RNA features (from 3_GeneExpress_extractfeatures.py)
###          - Patientinfo (columns: case, survival_months, vital_status (1 or 0))
### - output:  Concatenated FFPE and RNA features for Early Fusion model
###############################################################################
###############################################################################
### Example command
### $ 1_Concat2Features.py
###################################################
###################################################

### Set Environment
####################
import pandas as pd
import numpy as np

### Concat Features
####################

### RNA
rna_cases = pd.read_csv('extractfeatures/rna_cases.csv', header=0)
rna_features = pd.read_csv('extractfeatures/rna_features.csv', header=None)
print(rna_cases.shape, rna_features.shape)

## Pathology
pathology_cases = pd.read_csv('extractfeatures/pathology_cases.csv', header=0)
pathology_features = pd.read_csv('extractfeatures/pathology_features.csv', header=None)
print(pathology_cases.shape, pathology_features.shape)

## Patient Info
#needed: 
# - patient id (= case_)
# - survival_months
# - vital_status (1 or 0)
patientinfo = pd.read_csv("patientinfo.csv", header=0)
patientinfo = patientinfo[['case', 'survival_months', 'vital_status']]
print(patientinfo.shape)

## Sanity check
pathology_cases_id = list(pathology_cases['0'])
rna_cases_id = list(rna_cases['0'])
len(set(rna_cases_id + pathology_cases_id))

## Merge
rna_df = rna_features
rna_df['case'] = rna_cases_id

pathology_df = pathology_features
pathology_df['case'] = pathology_cases_id

merged_df = rna_df.merge(pathology_df, how="inner", on='case')
print(merged_df.shape)

final_df = patientinfo.merge(merged_df, how="inner", on='case')
print(final_df.shape)

## Output
final_df.columns = ['case', 'survival_months', 'vital_status'] + ['feature_'+str(col)  for col in list(final_df.columns)[4:]]
final_df.to_csv('features.csv', index=False)