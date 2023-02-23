import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm

from mil.varmil import VarMIL
from ovca import OVCA, OVMIL
from models.allconv import AllConvNet
from models.kimianet_virtual import KimiaNet
from models.wrn50_virtual import WideResNet50_2

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Trains a VOS-OVCA Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['ovca'],                                # TODO
                    default='ovca',
                    help='')
parser.add_argument('--split_dir', type=str, default='', 
                    help='json split produced by create splits')
parser.add_argument('--histotypes', action='store', type=str, nargs="+", 
                    help='space separated str IDs specifying histotype labels')
parser.add_argument('--num_split', type=int, default=0,
                    help='index of current split, e.g. 0-2 for 3-fold cv')
parser.add_argument('--unfreeze_epoch', type=int, default=40,
                    help='epoch at which to unfreeze pretrained layers')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for torch, cuda, and np')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn', 'kimia'], help='Choose architecture.')
# parser.add_argument('--calibration', '-c', action='store_true',                                                    # don't need
#                     help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# energy reg
parser.add_argument('--start_epoch', type=int, default=40)
parser.add_argument('--sample_number', type=int, default=1000)
parser.add_argument('--select', type=int, default=1)
parser.add_argument('--sample_from', type=int, default=10000)
parser.add_argument('--loss_weight', type=float, default=0.1)



args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# computed from subset (1/6) of Ov_van
mean = [x / 255 for x in [207.79, 177.028, 209.72]]
std = [x / 255 for x in [28.79, 33.26, 18.41]]

# Create dataloaders
train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomVerticalFlip(), trn.RandomResizedCrop(512, scale=(0.9, 1.0), ratio=(0.95, 1.05)), # should use to avoid overfitting
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'ovca':
    labels = args.histotypes
    num_classes = len(labels)

    train_data = OVCA(args.split_dir, chunk=0, transform=train_transform, target_transform=None, idx_to_label=labels, key_word='Tumor')
    val_data = OVCA(args.split_dir, chunk=1, transform=test_transform, target_transform=None, idx_to_label=labels, key_word='Tumor')
    test_data = OVCA(args.split_dir, chunk=2, transform=test_transform, target_transform=None, idx_to_label=labels, key_word='Tumor')

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=False, # TODO 
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False, 
    num_workers=args.prefetch, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'allconv':
    net = AllConvNet(num_classes)
    num_channels = 0 # TODO
elif args.model == 'kimia':
    net = KimiaNet(num_classes)
    num_channels = net.model.classifier.in_features # embedding dimension
else:
    net = WideResNet50_2(num_classes=num_classes, freeze=True)
    num_channels = net.model.fc.in_features # embedding dimension

# start_epoch = 0 ???
split_indicator = str(args.num_split)

# TODO: restore model is desired

# Initialize GPU
# TODO: 
# import torch.distributed as dist6
# import torch.multiprocessing as mp
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(args.seed)

# TODO
cudnn.benchmark = True  # find algorithms for best runtime

# Initialize VOS components = weight_energy and logistic_regression <-- extract? 
weight_energy = torch.nn.Linear(num_classes, 1).cuda()
torch.nn.init.uniform_(weight_energy.weight)
data_dict = torch.zeros(num_classes, args.sample_number, num_channels).cuda()
number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
eye_matrix = torch.eye(num_channels, device='cuda') # was 128, num_channel replacement here is correct
logistic_regression = torch.nn.Linear(1, 2)
logistic_regression = logistic_regression.cuda()

# Initialize optim
optimizer = torch.optim.SGD(
    list(net.parameters()) + list(weight_energy.parameters()) + \
    list(logistic_regression.parameters()), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    Requires: global var weight_energy

    value.exp().sum(dim, keepdim).log()
    """
    import math
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)


'''
!!! Debugging 
'''
# sample = next(iter(train_loader))

# Training
def train(epoch):
    net.train()  # enter train mode
    loss_avg = 0.0
    i = 0
    for data, target in train_loader:
        '''
        !!! 
        '''
        i += 1
        # data, target = sample
        data, target = data.cuda(), target.cuda()


        # forward
        out_dict = net.forward(data)
        x, output = out_dict['x'], out_dict['output']

        # energy regularization.
        '''
        data_dict = queue
        number_dict = length of queue for each class
        '''
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
            # maintaining an ID data queue for each class.
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                      output[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * args.sample_number and epoch >= args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                      output[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix


            for index in range(num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((args.sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, args.select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
            if len(ood_samples) != 0:
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = log_sum_exp(x, 1)
                if args.model == 'kimia':
                    predictions_ood = net.model.classifier(ood_samples)
                elif args.model == 'wrn':
                    predictions_ood = net.model.fc(ood_samples)
                else: # args.model == 'allconv
                    predictions_ood = net.fc(ood_samples) 
                # TODO: not good!!!

                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = log_sum_exp(predictions_ood, 1)

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                           torch.zeros(len(ood_samples)).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())

                if epoch % 10 == 0:
                    print(f"lr_reg_loss for id ood classification: {lr_reg_loss}")
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        # breakpoint()
        loss += args.loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        if i > 50:
            print(f"loss_avg {loss_avg}")
            break

    state['train_loss'] = loss_avg




if __name__ == '__main__':
    for epoch in range(args.epochs):
        # print(args.epochs)
        train(epoch)
        if epoch % 10 == 0:
            print(f"epoch: {epoch}")
        if epoch == args.unfreeze_epoch:
            print("unfreezed!")
        if epoch == args.start_epoch:
            print("sampling started!")