import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

class KimiaNet(nn.Module):
    def __init__(self, num_classes=5, freeze=True, state_dict_path='/projects/ovcare/classification/cshi/OoD/vos/classification/CIFAR/snapshots/baseline/KimiaNetPyTorchWeights_new.pth'):
        '''
        load weights & customize final layer
        '''
        super(KimiaNet, self).__init__()
        self.model = torchvision.models.densenet121(num_classes=30)
        state_dict = torch.load(state_dict_path, map_location = torch.device('cpu'))
        self.model.load_state_dict(state_dict)

        num_ftrs = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(num_ftrs, num_classes)

        nn.init.kaiming_normal_(self.model.classifier.weight.data, nonlinearity="relu")
        self.model.classifier.bias.data = torch.zeros(size=self.model.classifier.bias.size()).cuda()

        if freeze:
            for name, param in self.model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
    
    def forward(self, x):
        # print(type(x))
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return {"x": self.model.classifier(out), "output": out}