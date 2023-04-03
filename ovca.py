import os
import glob
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
    dataset             root_dir of image embeddings as .pt files. Path structure: 'root_dir/class_label/*.pt'
    histotypes          list of histotype labels
    label_to_idx        reverse mapping of histotypes
    '''
    def __init__(self, dataset, histotypes, debug=False): 
        self.dataset = dataset
        self.histotypes = histotypes
        self.debug = debug
        self.label_to_idx = {v: k for k, v in enumerate(histotypes)}
        self.images, self.labels = self._find_pt_files(dataset)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx): 
        label = self.labels[idx]
        image = torch.load(self.images[idx]) 
        return image, label
        
    def _find_pt_files(self, root_dir):
        pt_files = []
        labels = []
        for class_label in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_label)
            image_embs = glob.glob(f"{class_path}/*.pt")
            pt_files.extend(image_embs)
            labels.extend([self.label_to_idx[class_label]] * len(image_embs))

        if self.debug:
            pdb.set_trace()
            x = torch.load(pt_files[0])
            print(f"Found {len(pt_files)} pt files in {root_dir} with {label[:5]} histotypes")
            print(f"{x.sum()}, {x.shape}, {type(x)}")
        return pt_files, labels

class OVEM(Dataset):
    def __init__(self, dataset, histotypes, debug=False, ood=False):
        self.dataset = dataset
        self.histotypes = histotypes
        self.debug = debug
        self.label_to_idx = {v: k for k, v in enumerate(histotypes)}
        self.images, self.labels = self._find_pt_files(dataset)
        self.scores = [path.replace("embeddings_150", "scores_150") for path in self.images] # list of path to scores
        if ood:
            self.scores = [path.replace("embeddings_150", "scores_rare") for path in self.images] # list of path to scores
        if debug:
            print("entered OVEM")
            print(self.scores[:5])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = torch.load(self.images[idx]).cpu()
        image = image[~torch.all(image == 0, dim=1)] # remove zero rows
        score = self.scores[idx]
        return image, label, score

    def _find_pt_files(self, root_dir):
        pt_files = []
        labels = []
        for class_label in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_label)
            image_embs = glob.glob(f"{class_path}/*.pt")
            pt_files.extend(image_embs)
            labels.extend([self.label_to_idx[class_label]] * len(image_embs))

        if self.debug:
            # pdb.set_trace()
            x = torch.load(pt_files[0])
            print(f"Found {len(pt_files)} pt files in {root_dir} with {labels[:5]} histotypes")
            print(f"{x.sum()}, {x.shape}, {type(x)}")
        return pt_files, labels

    def get_images(self):
        return self.images

class OVSCORE(Dataset):
    def __init__(self, dataset, debug=False):
        self.dataset = dataset
        self.scores = self._find_score_files(dataset)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        score = torch.load(self.scores[idx])
        return score
    
    def _find_score_files(self, dataset):
        scores = []
        for class_label in os.listdir(dataset):
            class_path = os.path.join(dataset, class_label)
            image_scores = glob.glob(f"{class_path}/*.pt")
            scores.extend(image_scores)
        if self.debug:
            # pdb.set_trace()
            x = torch.load(scores[0])
            print(f"Found {len(scores)} score files in {dataset}")
            print(f"{x.sum()}, {x.shape}, {type(x)}")
        return scores