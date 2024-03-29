import numpy as np
import os
import re
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm

import pdb
from mil.varmil import VarMIL
from ovca import OVCA, OVMIL
from utils_ovca import class_proportion, preprocess_load
from models.allconv import AllConvNet
from models.kimianet_virtual import KimiaNet
from models.densenet import DenseNet3
from models.wrn50_virtual import WideResNet50_2

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

# Dataset
parser = argparse.ArgumentParser(description='Trains a VOS-OVCA Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['ovca'],
                    default='ovca',
                    help='')
parser.add_argument('--split_dir', type=str, default='', 
                    help='json split produced by create splits')
parser.add_argument('--histotypes', action='store', type=str, nargs="+", 
                    help='space separated str IDs specifying histotype labels')
parser.add_argument('--num_split', type=int, default=0,
                    help='index of current split, e.g. 0-2 for 3-fold cv')
parser.add_argument('--freeze', '-f', action='store_true', 
                    help='freeze convolutional layers during training')
parser.add_argument('--unfreeze_epoch', type=int, default=40,
                    help='epoch at which to unfreeze pretrained layers')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for torch, cuda, and np')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn', 'kimia', 'dense'], help='Choose architecture.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/ovca', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--save_freq', type=int, default=1, help='Save frequency.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# energy reg
parser.add_argument('--start_epoch', type=int, default=40)
parser.add_argument('--sample_number', type=int, default=1000)
parser.add_argument('--select', type=int, default=1)
parser.add_argument('--sample_from', type=int, default=10000)
parser.add_argument('--loss_weight', type=float, default=0.1)
parser.add_argument('--threshold', type=float, default=0.05)


args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

mean = [x / 255 for x in [207.79, 177.028, 209.72]]
std = [x / 255 for x in [28.79, 33.26, 18.41]]

# Create dataloaders
train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomVerticalFlip(), trn.RandomResizedCrop(512, scale=(0.9, 1.0), ratio=(0.95, 1.05)), # should use to avoid overfitting
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'ovca':
    labels = args.histotypes
    num_classes = len(labels)
    class_weights = class_proportion(args.split_dir, args.histotypes).to('cuda')

    train_data = OVCA(args.split_dir, chunk=0, train_transform=test_transform, target_transform=None, test_transform=test_transform, idx_to_label=labels, key_word='Tumor') # TODO
    val_data = OVCA(args.split_dir, chunk=1, test_transform=test_transform, target_transform=None, idx_to_label=labels, key_word='Tumor')
    test_data = OVCA(args.split_dir, chunk=2, test_transform=test_transform, target_transform=None, idx_to_label=labels, key_word='Tumor')

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
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
    net = KimiaNet(num_classes, freeze=args.freeze)
    num_channels = net.emb_dim # embedding dimension
elif args.model == 'dense':
    net = DenseNet3(100, num_classes)
    num_channels = net.in_planes # embedding dimension
elif args.model == 'wrn':
    net = WideResNet50_2(num_classes=num_classes, freeze=args.freeze)
    num_channels = net.emb_dim # embedding dimension

split_indicator = '_' + str(args.num_split)


if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(args.seed)

# restore model 
start_epoch = 0
if not args.load == '':  
    net.load_state_dict(preprocess_load(args.load, args.ngpu))
    temp = str.split(args.load, '_')[-1]
    i = re.sub("[^0-9]", "", temp)
    print('Model restored! Epoch:', i)
    start_epoch = int(i) + 1

cudnn.benchmark = True  

# Initialize VOS components = weight_energy and logistic_regression
weight_energy = torch.nn.Linear(num_classes, 1).cuda()
torch.nn.init.uniform_(weight_energy.weight)
data_dict = torch.zeros(num_classes, args.sample_number, num_channels).cuda()
number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
eye_matrix = torch.eye(num_channels, device='cuda') 
logistic_regression = torch.nn.Linear(1, 2)
logistic_regression = logistic_regression.cuda()

# Initialize optim
optimizer = torch.optim.Adam(
    list(net.parameters()) + list(weight_energy.parameters()) + \
    list(logistic_regression.parameters()), state['learning_rate'], #momentum=state['momentum'],
    weight_decay=state['decay'], #nesterov=True
    )

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

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

# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output, _ = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

# validate function
def val():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output, _ = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['val_loss'] = loss_avg / len(val_loader)
    state['val_accuracy'] = correct / len(val_loader.dataset)


# Training
def train(epoch):
    # unfreeze training if frozen
    if args.freeze and epoch >= args.unfreeze_epoch:
        for name, param in net.module.model.named_parameters():
                param.requires_grad = True

    net.train()  
    loss_avg = 0.0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net(data)

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
            if torch.isnan(data_dict).any():
                print("nan in data_dict!")
                pdb.set_trace()
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

            if torch.isnan(temp_precision).any():
                print("nan in temp_precision!")
                pdb.set_trace()
            if torch.isnan(mean_embed_id).any():
                print("nan in means!")
                pdb.set_trace()

            for index in range(num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((args.sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                
                # breakpoint()
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, args.select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                
            if len(ood_samples) != 0:
                energy_score_for_fg = log_sum_exp(x, 1)
                if torch.isnan(energy_score_for_fg).any(): # debug
                    print("nan in energy_score_for_fg!")
                    pdb.set_trace()
                if args.model == 'kimia':
                    predictions_ood = net.module.model.classifier(ood_samples)
                elif args.model == 'wrn':
                    predictions_ood = net.model.fc(ood_samples) # TODO: ngpu = 1
                else: # args.model == allconv cifar_dense or cifar_wrn
                    predictions_ood = net.fc(ood_samples) 
                # TODO: fix the  switch on type

                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = log_sum_exp(predictions_ood, 1)
                if torch.isnan(energy_score_for_bg).any(): # debug
                    print(f"nan in energy_score_for_bg! epoch: {epoch}")
                    pdb.set_trace()

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                           torch.zeros(len(ood_samples)).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())

                if torch.isnan(lr_reg_loss):
                    print(f"nan! energy_score: {energy_score_for_bg}, {energy_score_for_fg}, i: {i}, epoch: {epoch}")
                    pdb.set_trace()

        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        # backward
        optimizer.zero_grad()

        loss = F.cross_entropy(x, target, weight=class_weights)
        loss += args.loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    state['reg_loss'] = float(loss)
    # state['avg_log_prob'] = np.mean(prob) # TODO
    # state['std_log_prob'] = np.std(prob)
    




if __name__ == '__main__':

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
    with open(os.path.join(args.save, 'training_results.csv'), 'w') as f:
        f.write('epoch,time(s),train_loss,reg_loss,val_loss,val_error(%)\n')


    for epoch in range(start_epoch, args.epochs):
        
        begin_epoch = time.time()

        train(epoch)
        val()

        if args.model == 'dense': # TODO adjust lr
            if epoch == 20:
                optimizer.param_groups[0]['lr'] *= args.learning_rate * 0.1
            elif epoch == 30:
                optimizer.param_groups[0]['lr'] *= args.learning_rate * 0.01
            elif epoch == 40:
                optimizer.param_groups[0]['lr'] *= args.learning_rate * 0.001

        # Save models
        if epoch % args.save_freq == 0:
            torch.save(net.state_dict(),
                    os.path.join(args.save, f'net_{epoch}.pt'))
            torch.save(weight_energy.state_dict(), os.path.join(args.save, f'weight_energy_{epoch}.pt'))
            torch.save(logistic_regression.state_dict(), os.path.join(args.save, f'logistic_regression_{epoch}.pt'))

        # Let us not waste space and delete the previous model
        # prev_path = os.path.join(args.save, args.dataset + split_indicator + '_' + args.model +
        #                         '_' + str(args.loss_weight) + \
        #                         '_' + str(args.sample_number)+ '_' + str(args.start_epoch) + '_' +\
        #                         str(args.select) + '_' + str(args.sample_from)  + '_' + 'epoch_' + str(epoch - 1) + '.pt')
        # if os.path.exists(prev_path): os.remove(prev_path)

        # Show results
        with open(os.path.join(args.save, 'training_results.csv'), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['reg_loss'],
                state['val_loss'],
                100 - (100. * state['val_accuracy']),
            ))

        # # print state with rounded decimals
        # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Reg Loss {3:.3f} | Val Loss {4:.3f} | Val Error {5:.2f}'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state['train_loss'],
            state['reg_loss'],
            state['val_loss'],
            100 - (100. * state['val_accuracy']))
        )
