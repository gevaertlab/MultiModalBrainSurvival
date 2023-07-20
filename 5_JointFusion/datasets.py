import os

import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torch
import pandas as pd

from torchvision import transforms

class PatchBagDataset(Dataset):

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
            WSI = row['wsi_file_name'].split('.')[0]
            n_patches = sum(1 for _ in open(os.path.join(self.patch_data_path, WSI, 'loc.txt'))) - 2
            n_patches = min(n_patches,self.max_patches_total)
            images = [os.path.join(self.patch_data_path, WSI, WSI + "_patch_{}.png".format(i)) for i in
                      range(n_patches)]
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
        result = row
        result['patch_bag'] = img

        return result
    
class PatchBagRNADataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None, bag_size=40, max_patch_per_wsi=400):

        self._patch_data_path = patch_data_path
        self._csv_path = csv_path
        self._img_size = img_size
        self._transforms = transforms
        self.bag_size = bag_size
        self._max_patch_per_wsi = max_patch_per_wsi
        self.data = {}
        self.index=[]
        self._preprocess()

    def _preprocess(self):

        self.data, self.index = get_data_rna_bag_histopathology(self._csv_path,self._patch_data_path,self._max_patch_per_wsi, self.bag_size)

    def __len__(self):

        return len(self.index)

    def __getitem__(self, idx):

        (WSI, i) = self.index[idx]
        imgs=[]
        item = self.data[WSI].copy()
        for patch in item['images'][i:i + self.bag_size]:
            with open(patch, "rb") as f:
                img = Image.open(f).convert('RGB')
            if self._transforms is not None:
                img = self._transforms(img)
            imgs.append(img)
        img=torch.stack(imgs,dim=0)
        item['patch_bag'] = img

        return item

def get_data_rna_bag_histopathology(csv_path, patch_path, limit, bag_size):

    dataset = {}
    index=[]
    data = pd.read_csv(csv_path)

    for _, row in data.iterrows():
        
        WSI = row['wsi_file_name']

        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        rna_data = torch.tensor(rna_data, dtype=torch.float32)

        row = row[[x for x in row.keys() if 'rna_' not in x]].to_dict()
        row['rna_data'] = rna_data

        row['vital_status'] = np.float32(row['vital_status'])
        row['survival_months'] = np.float32(row['survival_months'])
        row['survival_bin'] = np.long(row['survival_bin'])

        n_patches = sum(1 for _ in open(os.path.join(patch_path, WSI, 'loc.txt'))) - 2
        images = [os.path.join(patch_path, WSI, WSI + "_patch_{}.png".format(i)) for i in
                  range(n_patches)]

        if limit is not None:
            images = images[:limit]   

        row['WSI']=WSI
        row['images']=images
        row['n_images']=len(images)       
        
        dataset[WSI] = {w.lower(): row[w] for w in row.keys()}
                       
        for k in range(len(images) // bag_size):
            index.append((WSI, bag_size * k))
       
    return dataset, index

class PatchRNADataset(Dataset):

    def __init__(self, patch_data_path, csv_path, img_size, transforms=None, max_patch_per_wsi=400):

        self._patch_data_path = patch_data_path
        self._csv_path = csv_path
        self._img_size = img_size
        self._transforms = transforms
        self._max_patch_per_wsi = max_patch_per_wsi
        self.data = None
        self._preprocess()

    def _preprocess(self):

        self.data = get_data_rna_histopathology(self._csv_path,self._patch_data_path,limit=self._max_patch_per_wsi)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx].copy()
        patch = item['patch']
        img = Image.open(patch).convert('RGB')
        if self._transforms is not None:
            img = self._transforms(img)
        item['img'] = img
        return item
    
def get_data_rna_histopathology(csv_path, patch_path, limit=None):

    dataset = []
    data = pd.read_csv(csv_path)

    for _, row in data.iterrows():
        
        WSI = row['wsi_file_name']
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        rna_data = torch.tensor(rna_data, dtype=torch.float32)

        row = row[[x for x in row.keys() if 'rna_' not in x]].to_dict()
        row['patch_folder'] = WSI
        row['rna_data'] = rna_data

        row['vital_status'] = np.float32(row['vital_status'])
        row['survival_months'] = np.float32(row['survival_months'])
        row['survival_bin'] = np.long(row['survival_bin'])

        n_patches = sum(1 for _ in open(os.path.join(patch_path, WSI, 'loc.txt'))) - 2
        images = [os.path.join(patch_path, WSI, WSI + "_patch_{}.png".format(i)) for i in
                  range(n_patches)]

        if limit is not None:
            
            images = images[:limit]
            
        for i in images:

            item = row.copy()
            item['patch'] = os.path.join(patch_path, patch_folder, i)
            dataset.append(item)

    return dataset

