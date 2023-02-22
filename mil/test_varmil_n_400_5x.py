import torch
import os
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import dgl
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from scipy import sparse
from dgl.data.utils import load_graphs
import dgl.nn.pytorch as dglnn
from dgl.nn import GATConv
import torch.nn as nn
from scipy.linalg import block_diag
import dgl.nn.pytorch as dglnn
from dgl.nn import GATConv
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score as bas
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(256)
random_seeds = torch.randint(0, 10000, (10,)).numpy()
splits = ['split.1_2_train_3_eval', 'split.1_3_train_2_eval', 'split.2_3_train_1_eval']
sets = ['train', 'val', 'test']
#root = "/projects/ovcare/classification/Ali/Heram/Dataset/kimianet_embeddings/"
graph_path = "/projects/ovcare/classification/Ali/Heram/Dataset/kimianet_embeddings/original_kimianet_graphs_n_400/"
mag_levels = ['5x', '10x', '20x']
path_manifest = "/projects/ovcare/classification/Ali/Heram/Dataset/manifest/auto_annotate_manifest.csv"
path_csv = {}

path_csv['test'] = "/projects/ovcare/classification/Ali/Heram/Dataset/Random_patches_for_kimianet/cross_validation_splits/5x_test_"
path_csv['train'] = "/projects/ovcare/classification/Ali/Heram/Dataset/Random_patches_for_kimianet/cross_validation_splits/5x_train_"
path_csv['val'] = "/projects/ovcare/classification/Ali/Heram/Dataset/Random_patches_for_kimianet/cross_validation_splits/5x_val_"
group = {}
manifest = pd.read_csv(path_manifest)

n = 400

from torch.utils.data import Dataset
class heram_dataset(Dataset):
    def __init__(self, data_dict, root_dir, transform=None):
        self.data_dict=data_dict
        self.root_dir=root_dir
        self.transform=transform
        
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        graph = self.data_dict[idx][0][0]
        label = graph.ndata['label'][0]
        patient_name = self.data_dict[idx][1]
        #len_slides = len(self.data_dict[idx][2])
        
        if self.transform:
            graph = self.transform(graph)
        
        return(graph,label)

dictionary = {}
for split in splits:
    print(split)
    dictionary[split]= {}
    for set_ in sets:
        dataset = pd.read_csv(path_csv[set_] + split + '.csv')
        slides = dataset.name
        slides = list(dict.fromkeys(slides))
        print(set_)
        dictionary[split][set_]=[]
        path_to_graph = graph_path + split + '/' + set_ + '/'
        for wsi in slides:
            #if not(wsi in ['EQJ20', 'W5XAP', 'XBQA7','BF5S6']):
            graph = load_graphs(path_to_graph + 'graph_' + wsi + '.bin')
            dictionary[split][set_].append((graph[0], wsi))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset

import dgl.nn.pytorch as dglnn
from dgl.nn import GATConv
import torch.nn as nn

class VarMIL(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        dim = 128
        torch.autograd.set_detect_anomaly(True)
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention  = nn.Sequential(nn.Linear(in_size, dim),
                                       nn.Tanh(),
                                       nn.Linear(dim, 1))
        self.classifier = nn.Sequential(nn.Linear(2*in_size, dim),
                                       nn.ReLU(),
                                       nn.Linear(dim, 5))

    def forward(self, x):
        """
        x   (input)            : B (batch size) x K (nb_patch) x out_channel
        A   (attention weights): B (batch size) x K (nb_patch) x 1
        M   (weighted mean)    : B (batch size) x out_channel
        S   (std)              : B (batch size) x K (nb_patch) x out_channel
        V   (weighted variance): B (batch size) x out_channel
        nb_patch (nb of patch) : B (batch size)
        M_V (concate M and V)  : B (batch size) x 2*out_channel
        out (final output)     : B (batch size) x num_classes
        """
        b, k, c = x.shape
        A = self.attention(x)
        A = A.masked_fill((x == 0).all(dim=2).reshape(A.shape), -9e15) # filter padded rows
        A = F.softmax(A, dim=1)                                        # softmax over K
        M = torch.einsum('b k d, b k o -> b o', A, x)                  # d is 1 here
        S = torch.pow(x-M.reshape(b,1,c), 2)
        V = torch.einsum('b k d, b k o -> b o', A, S)
        nb_patch = (torch.tensor(k).expand(b)).to(self.device)
        nb_patch = nb_patch - torch.sum((x == 0).all(dim=2), dim=1)    # filter padded rows
        nb_patch = nb_patch / (nb_patch - 1)                           # I / I-1
        nb_patch = torch.nan_to_num(nb_patch, posinf=1)                # for cases, when we have only 1 patch (inf)
        V = V * nb_patch[:, None]                                      # broadcasting
        M_V = torch.cat((M, V), dim=1)
        out = self.classifier(M_V)
        return out

import copy
from dgl.dataloading import GraphDataLoader
path_to_save = "/projects/ovcare/classification/Ali/Heram/codes/400_patches_kimianet/varmil/weights/"

for r_seed in random_seeds:
    test_preds = {'split.1_2_train_3_eval':[], 'split.1_3_train_2_eval':[], 'split.2_3_train_1_eval':[]}
    test_labels = {'split.1_2_train_3_eval':[], 'split.1_3_train_2_eval':[], 'split.2_3_train_1_eval':[]}
    print('seed:', r_seed, flush=True)
    torch.manual_seed(r_seed)
    for split in splits:
        train_loader = heram_dataset(dictionary[split]['train'], '/')
        val_loader = heram_dataset(dictionary[split]['val'], '/')
        test_loader = heram_dataset(dictionary[split]['test'], '/')
        tmp = []
        weight = []
        for graph in train_loader:
            tmp.append(np.array(graph[1]))
        for label in range(5):
            weight.append((np.array(tmp)==label).sum())
        weights = np.array(weight, np.float32)**(-1)*min(weight)
        batch_size = 1
        dataloader_train = GraphDataLoader(
            train_loader,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True)

        dataloader_val = GraphDataLoader(
            val_loader,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False)

        dataloader_test = GraphDataLoader(
            test_loader,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False)
        
        #import torch.nn.functional as F
        
        # Only an example, 7 is the input feature size
        model = VarMIL(1024)
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
        #best_model = model
        k = 0
        hist_val = []
        running_corrects_test = 0
        pre_best = 0
        model.cuda()
        for epoch in range(100):
            running_corrects = 0;
            running_corrects_val = 0;
            for batched_graph, labels in dataloader_train:
                k = k +1 
                #print('batch:', k)
                feats = batched_graph.ndata['x']
                feats = feats.reshape(batch_size,feats.shape[0],1024)
                gap = int(feats.shape[1]/3)
                feats = feats[:,0*gap:1*gap]
                feats = feats.to(device)
                logits = model(feats)
                labels = labels.to(device)
                _, preds = torch.max(logits, 1)
                running_corrects += torch.sum(preds == labels)
                #print(running_corrects)
                loss = F.cross_entropy(logits, labels, weight=torch.tensor(weights, device=device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            epoch_acc = running_corrects.double() / len(train_loader)
            #print('epoch:', epoch, '----loss:', loss, '----acc:', epoch_acc)
            model.eval()

            preds_hist = []
            labels_hist = []

            for batched_graph, labels in dataloader_val:
                feats = batched_graph.ndata['x']
                feats = feats.reshape(batch_size,feats.shape[0],1024)
                gap = int(feats.shape[1]/3)
                feats = feats[:,0*gap:1*gap]
                feats = feats.to(device)
                logits = model(feats)
                labels = labels.to(device)
                _, preds = torch.max(logits, 1)
                running_corrects_val += torch.sum(preds == labels)
                preds_hist.extend(preds.cpu().numpy())
                labels_hist.extend(labels.cpu().numpy())
                #print(running_corrects_val)
                loss = F.cross_entropy(logits, labels)

            #val_acc = running_corrects_val.double() / len(val_loader)
            val_acc = bas(labels_hist, preds_hist)
            if val_acc > pre_best:
                pre_best = val_acc
                best_model_w = copy.deepcopy(model.state_dict())
            hist_val.append(val_acc)
            print('epoch:', epoch, '----loss:', loss.item(), '----acc:',val_acc.item(), flush = True)
            model.train()
        print('max val acc:', pre_best.item(), flush = True)
        
        #model.eval()
        best_model = model
        best_model.load_state_dict(best_model_w)
        #best_model.eval()
        best_model.cuda()
        for batched_graph, labels in dataloader_test:
            feats = batched_graph.ndata['x']
            feats = feats.reshape(batch_size,feats.shape[0],1024)
            gap = int(feats.shape[1]/3)
            feats = feats[:,0*gap:1*gap]
            feats = feats.to(device)
            labels = labels.to(device)
            logits = best_model(feats)
            _, preds = torch.max(logits, 1)
            running_corrects_test += torch.sum(preds == labels)
            test_preds[split].extend(np.array(torch.tensor(preds, device = 'cpu')))
            test_labels[split].extend(np.array(torch.tensor(labels, device= 'cpu')))
            #print(running_corrects_val)
            loss = F.cross_entropy(logits, labels)
        test_acc = running_corrects_test.double() / len(test_loader)
        torch.save(best_model.state_dict(), path_to_save + 'r_seeds/5x/' + str(r_seed) + split + '_' + 'varpmil_bacc_5x' + '.pt' )
        print('test acc:', test_acc.item(), flush = True)

    np.save(path_to_save + 'r_seeds/5x/' + str(r_seed) + '_bacc_test_preds_5x.npy', test_preds)
    np.save(path_to_save + 'r_seeds/5x/' + str(r_seed) + '_bacc_test_labels_5x.npy', test_labels)