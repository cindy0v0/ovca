import torch
import torch.nn as nn
import torchvision.models as models

class WideResNet50_2(nn.Module):
    def __init__(self, num_classes, freeze=True):
        super(WideResNet50_2, self).__init__()
        self.model = models.wide_resnet50_2(weights='DEFAULT')
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

        nn.init.kaiming_normal_(self.model.fc.weight.data, nonlinearity="relu")
        self.model.fc.bias.data = torch.zeros(size=self.model.fc.bias.size()).cuda()

        if freeze:
            for name, param in self.model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False

    def forward(self, x):
        # print(f"x shape: {x.shape}")
        x = x.type(torch.cuda.FloatTensor)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(f"x shape before fc: {x.shape}")
        return {"x": self.model.fc(x), "output": x}
