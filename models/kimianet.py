import torch
import torch.nn as nn
import torchvision

class KimiaNet(nn.Module):
    def __init__(self, num_classes):
        super(KimiaNet, self).__init__()
        self.model = torchvision.models.densenet121()
        
        self.model.load_state_dict(torch.load('/projects/ovcare/classification/Ali/Ovarian_project/Pytorch_Codes/KimiaNet/KimiaNetPyTorchWeights.pth'))

