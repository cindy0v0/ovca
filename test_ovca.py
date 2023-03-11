import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
# from models.wrn import WideResNet
# from models.densenet import DenseNet3
from models.wrn_virtual import WideResNet
from models.densenet import DenseNet3
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage

from models.kimianet_virtual import KimiaNet
from models.wrn50_virtual import WideResNet50_2
from ovca import OVCA, OV_OOD

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    import utils.score_calculation as lib
    from utils_ovca import preprocess_load

parser = argparse.ArgumentParser(description='Evaluates a OVCA OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='ovca', help='Method name.')
# OVCA
parser.add_argument('--split_dir', type=str, default='', 
                    help='json split produced by create splits')
parser.add_argument('--histotypes', action='store', type=str, nargs="+", 
                    help='space separated str IDs specifying histotype labels')
parser.add_argument('--num_split', type=int, default=0,
                    help='index of current split, e.g. 0-2 for 3-fold cv')
parser.add_argument('--ood_dir', type=str, default='', help='root directory to rare ov patches by subtype')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--drop_rate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--model_name', default='kimia', type=str)

args = parser.parse_args()
print(args)
# torch.manual_seed(1)
# np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [207.79, 177.028, 209.72]]
std = [x / 255 for x in [28.79, 33.26, 18.41]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'ovca' in args.method_name:
    labels = args.histotypes
    num_classes = len(labels)
    test_data = OVCA(args.split_dir, chunk=2, test_transform=test_transform, target_transform=None, idx_to_label=labels, key_word='Tumor')
    # ood_data = OV_OOD(args.ood)


test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
# Create model
if args.model_name == 'res':
    net = WideResNet50_2(num_classes=num_classes, drop_rate=args.drop_rate, freeze=False)
else: 
    net = KimiaNet(num_classes=num_classes, drop_rate=args.drop_rate, freeze=False)
    

start_epoch = 0

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

# Restore model
if args.load != '':
    if os.path.isfile(args.load):
        state_dict = preprocess_load(args.load, args.ngpu) # TODO: change state_dict 
        net.load_state_dict(state_dict)
        print('Model restored!')
    else:
        assert False, "could not resume "+ args.load

net.eval()

# torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

# ood_num_examples = len(ood_data)
# expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

num_examples = 0 # TODO: does not support M

def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.cuda()

            output, _ = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':
                    _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
                else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()

        
if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, args.test_bs, args.T, args.noise, in_dist=True)
elif args.score == 'M':
    from torch.autograd import Variable
    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    if 'ovca' in args.method_name:
        train_data = OVCA(args.split_dir, chunk=0, target_transform=None, test_transform=test_transform, idx_to_label=labels, key_word='Tumor') # TODO
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False, 
                                          num_workers=args.prefetch, pin_memory=True)
    num_batches = num_examples // args.test_bs # TODOs

    temp_x = torch.rand(2,3,512,512) # ??? TODO
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = net.feature_list(temp_x)[1] # TODO
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader) 
    in_score = lib.get_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches, in_dist=True)
    print(in_score[-3:], in_score[-103:-100])
else:
    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('ID Error Rate {:.3f} %'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

# print('\nUsing OVCA as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.method_name)



# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list, acc_list = [], [], [], []

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs, acc = [], [], [], []

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches)
        else:
            out_score = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(in_score[:3], out_score[:3]) # TODO: change into mean and std of all energy
    print(np.mean(in_score), np.mean(out_score))
    print(np.std(in_score), np.std(out_score))
    
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs); acc = np.mean(acc)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr); acc_list.append(acc)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)


subdirs = os.listdir(args.ood_dir)
type_to_idx = {v: k + num_classes for k, v in enumerate(subdirs)} # TODO: coupling? 
# ood_dir = [os.path.join(args.ood_dir, x) for x in ood]
for subdir in subdirs:
    path = os.path.join(args.ood_dir, subdir)
    
    print(f'\n{subdir} Detection')
    ood_data = OV_OOD(path, type_to_idx[subdir], transform=test_transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    get_and_print_results(ood_loader)


print('\n\nMean Test Results!!!!!')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)






# /////////////// OOD Detection of Validation Distributions ///////////////

if args.validate is False:
    exit()

auroc_list, aupr_list, fpr_list = [], [], []

# /////////////// Uniform Noise ///////////////

dummy_targets = torch.ones(num_examples * args.num_to_avg)
ood_data = torch.from_numpy(
    np.random.uniform(size=(num_examples * args.num_to_avg, 3, 32, 32),
                      low=-1.0, high=1.0).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nUniform[-1,1] Noise Detection')
get_and_print_results(ood_loader)


# /////////////// Arithmetic Mean of Images ///////////////

if 'cifar10_' in args.method_name:
    ood_data = dset.CIFAR100('/nobackup-slow/dataset/cifarpy', train=False, transform=test_transform)
else:
    ood_data = dset.CIFAR10('/nobackup-slow/dataset/cifarpy', train=False, transform=test_transform)


class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

    def __len__(self):
        return len(self.dataset)


ood_loader = torch.utils.data.DataLoader(AvgOfPair(ood_data),
                                         batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nArithmetic Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)


# /////////////// Geometric Mean of Images ///////////////

if 'cifar10_' in args.method_name:
    ood_data = dset.CIFAR100('/nobackup-slow/dataset/cifarpy', train=False, transform=trn.ToTensor())
else:
    ood_data = dset.CIFAR10('/nobackup-slow/dataset/cifarpy', train=False, transform=trn.ToTensor())


class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return trn.Normalize(mean, std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

    def __len__(self):
        return len(self.dataset)


ood_loader = torch.utils.data.DataLoader(
    GeomMeanOfPair(ood_data), batch_size=args.test_bs, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

print('\n\nGeometric Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)

# /////////////// Jigsaw Images ///////////////

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
               x[:, 16:, :16]), 2),
    torch.cat((x[:, 16:, 16:],
               torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
), 1)

ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), jigsaw, trn.Normalize(mean, std)])

print('\n\nJigsawed Images Detection')
get_and_print_results(ood_loader)

# /////////////// Speckled Images ///////////////

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), speckle, trn.Normalize(mean, std)])

print('\n\nSpeckle Noised Images Detection')
get_and_print_results(ood_loader)

# /////////////// Pixelated Images ///////////////

pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.BOX).resize((32, 32), PILImage.BOX)
ood_loader.dataset.transform = trn.Compose([pixelate, trn.ToTensor(), trn.Normalize(mean, std)])

print('\n\nPixelate Detection')
get_and_print_results(ood_loader)

# /////////////// RGB Ghosted/Shifted Images ///////////////

rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                 x[2:, :, :], x[0:1, :, :]), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), rgb_shift, trn.Normalize(mean, std)])

print('\n\nRGB Ghosted/Shifted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Inverted Images ///////////////

# not done on all channels to make image ood with higher probability
invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), invert, trn.Normalize(mean, std)])

print('\n\nInverted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Validation Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
