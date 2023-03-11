import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import re
import pdb
import random 

class OVCA(Dataset):
    def __init__(self, img_json, chunk, train_transform=None, target_transform=None, transform_prob=0.6, test_transform=None, idx_to_label=None, key_word='Tumor'):
        '''
        Class with images as a list of patch paths, labels as a list of int label indices, and dictionary idx_to_label. keyword Tumor is used to parse for histotypes in patch path
        stores self.images as [image paths]
        '''
        label_to_index = {v: k for k, v in enumerate(idx_to_label)}
        with open(img_json) as f:
            data = json.load(f)
            self.images = data['chunks'][chunk]['imgs']
            labels = [re.search(rf'/{key_word}/(.*?)/', s).group(1) for s in self.images]
            self.labels = [label_to_index[x] for x in labels]
        self.train_transform = train_transform
        self.target_transform = target_transform
        self.transform_prob = transform_prob
        self.test_transform = test_transform 
        self.label_names = idx_to_label
        self.train = chunk == 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = Image.open(self.images[idx])
        if self.train and random.random() > self.transform_prob and self.train_transform: # TODO: decouple
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class OV_OOD(Dataset):
    def __init__(self, patches_path, label, transform=None, target_transform=None):
        self.images = self._find_png_files(patches_path)
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.label
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _find_png_files(self, directory):
        png_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".png"):
                    png_files.append(os.path.abspath(os.path.join(root, file)))
        return png_files

class OVMIL(Dataset):
    '''
    init effects
    dataset             directory of .pt files, nb_patch x embedding_size, 1 per slide. Naming convention: patient_id, histotype <-- should prioritize batches from the same patient = ordered dir & shuffle=False
    batch_size          size to pad nb_patch if < batch_size
    label_to_idx        reverse mapping of histotypes

    getitem returns
    patches             padded .pt
    labels              parse from name
    '''
    def __init__(self, dataset, batch_size, histotypes): 
        return None

    def __len__(self):
        return 0

    def __getitem__(self, idx): 
        return None
        
