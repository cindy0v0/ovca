import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.optim as optim
import copy
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset
class Ovarian_Dataset_5x(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame=data_frame
        self.root_dir=root_dir
        self.transform=transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.data_frame.iloc[idx, 2]
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 3]
        
        if self.transform:
            image = self.transform(image)
            
        return(image,label)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.cuda()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #print(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    x, outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    #print(preds)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            else:
                train_acc_history.append(epoch_acc)
                

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

splits_names = ['split.1_2_train_3_eval', 'split.1_3_train_2_eval', 'split.2_3_train_1_eval']
for sname in splits_names:
    path_train = '/projects/ovcare/classification/Ali/Heram/Dataset/Random_patches_for_kimianet/cross_validation_splits/5x_train_' + sname + '.csv'
    data_frame_train = pd.read_csv(path_train)
    path_val = '/projects/ovcare/classification/Ali/Heram/Dataset/Random_patches_for_kimianet/cross_validation_splits/5x_val_' + sname + '.csv'
    data_frame_val = pd.read_csv(path_val)
    path_test = '/projects/ovcare/classification/Ali/Heram/Dataset/Random_patches_for_kimianet/cross_validation_splits/5x_test_' + sname + '.csv'
    data_frame_test = pd.read_csv(path_test)

    class_weights = []
    for i in range(5):
        class_weights.append((data_frame_train.class_label == i).sum()//100)
    min_class_weights = min(class_weights)
    for i in range(5):
        class_weights[i] = min_class_weights/class_weights[i]

    data_train = Ovarian_Dataset_5x(data_frame=data_frame_train,
                                  root_dir = '',
                                  transform = transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])
    )

    data_val = Ovarian_Dataset_5x(data_frame=data_frame_val,
                                  root_dir = '',
                                  transform = transforms.Compose([
                                      #transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])
    )

    num_classes = 5
    batch_size = 8
    num_epochs = 20


    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    classes = ('CC', 'EC', 'HGSC','LGSC','MC')

    class fully_connected(nn.Module):
	    """docstring for BottleNeck"""
	    def __init__(self, model, num_ftrs, num_classes):
		    super(fully_connected, self).__init__()
		    self.model = model
		    self.fc_4 = nn.Linear(num_ftrs,num_classes)

	    def forward(self, x):
		    x = self.model(x)
		    x = torch.flatten(x, 1)
		    out_1 = x
		    out_3 = self.fc_4(x)
		    return  out_1, out_3
    
    model = torchvision.models.densenet121(pretrained=True)
    #for param in model.parameters():
    #	param.requires_grad = False
    model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
    num_ftrs = model.classifier.in_features
    model = fully_connected(model.features, num_ftrs, 30)
    model = model.to(device)
    model = nn.DataParallel(model)
    params_to_update = []


    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device = device, dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(torch.load('/projects/ovcare/classification/Ali/Ovarian_project/Pytorch_Codes/KimiaNet/KimiaNetPyTorchWeights.pth'))

    model.module.fc_4= nn.Linear(1024, 5)
    for param in model.parameters():
        param.requires_grad = True
    #for param in model.module.model[0].denseblock4.parameters():
    #    param.requires_grad = True

    k = 0
    for param in model.parameters():
        k = k + 1*param.requires_grad
    print(k)

    model_ft, hist1, hist2 = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    path_model = '/projects/ovcare/classification/Ali/Heram/codes/400_patches_kimianet/model_weights/fully_retrained_5x_balanced'+ sname + '.pth'
    path_val_acc = '/projects/ovcare/classification/Ali/Heram/codes/400_patches_kimianet/model_weights/fully_val_acc_5x_balanced'+ sname + '.pt'
    path_train_acc = '/projects/ovcare/classification/Ali/Heram/codes/400_patches_kimianet/model_weights/fully_train_acc_5x_balanced'+ sname + '.pt'

    torch.save(model_ft.state_dict(), path_model)
    torch.save(hist1, path_val_acc)
    torch.save(hist2, path_train_acc)