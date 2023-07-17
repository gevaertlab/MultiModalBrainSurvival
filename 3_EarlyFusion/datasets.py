import os

import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torch

from torchvision import transforms
import pandas as pd

class featureDataset(Dataset):
    """
    """

    def __init__(self, csv_path):

        self._csv_path = csv_path
        self.data = None
        self._preprocess()

    def _preprocess(self):
        self.data = get_data_feature(self._csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        return item

def get_data_feature(csv_path):
    
    dataset = []
    data = pd.read_csv(csv_path)
    for _, row in data.iterrows():

        feature_data = row[[x for x in row.keys() if 'feature_' in x]].values.astype(np.float32)
        feature_data = torch.tensor(feature_data, dtype=torch.float32)

        row = row[[x for x in row.keys() if 'feature_' not in x]].to_dict()
        row['feature_data'] = feature_data

        row['vital_status'] = np.float32(row['vital_status'])
        row['survival_months'] = np.float32(row['survival_months'])
        row['survival_bin'] = np.float32(row['survival_bin'])

        item = row.copy()
        dataset.append(item)

    return dataset
