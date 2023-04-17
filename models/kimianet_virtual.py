import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import pdb

class KimiaNet(nn.Module):
    def __init__(self, num_classes=5, dim=None, freeze=False, drop_rate=0.0, state_dict_path='./data/baseline/DensePretrainedWeights.pth'):
        super(KimiaNet, self).__init__()
        # load pretrained weights on TCGA,which has 30 classes
        self.model = torchvision.models.densenet121(num_classes=30, drop_rate=drop_rate)
        state_dict = torch.load(state_dict_path, map_location = torch.device('cpu'))
        self.model.load_state_dict(state_dict)

        # set classifier to desired num of classes
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(num_ftrs, num_classes)
        self.emb_dim = num_ftrs
    
    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.model.classifier(out), out