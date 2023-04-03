import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from mil.varmil import VarMIL
from ovca import OVEM, OVMIL, OVSCORE
from models.kimianet_virtual import KimiaNet
from utils_ovca import preprocess_load
import pdb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def _pad_array_with_zeros(array, target_rows):
    current_rows = array.shape[0]
    padding_rows = target_rows - current_rows

    if padding_rows <= 0:
        return array

    pad_width = [(0, padding_rows)] + [(0, 0)] * (array.ndim - 1)
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
    
    return padded_array

def _get_patch_scores(net, loader, use_xent=False, score='energy', in_dist=False, T=1.0):
    # target = int histotype of slide <-- remove all padded patches
    # data = 1 x p x d embeddings < -- squeeze ? 
    # generate patch level scores
    _score = []
    _right_score = []
    _wrong_score = []
    net.eval()
    net = net.to(device)

    with torch.no_grad():
        for batch_idx, (data, target, score_pth) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            score_pth = score_pth.cpu()
            output = net.model.classifier(data) # !!! classifier is the last layer of the model
            smax = (F.softmax(output, dim=2)).data.cpu().numpy()

            if use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            elif score == 'energy':
                _score.append(-to_np((T * torch.logsumexp(output / T, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
            
            # store scores
            _padded_score = _pad_array_with_zeros(_score, target_rows=150)
            _padded_score = torch.from_numpy(_padded_score)
            os.makedirs(os.path.dirname(score_pth), exist_ok=True)
            torch.save(_padded_score, score_pth) # TODO: debug

        if in_dist:
            return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
        else:
            return concat(_score).copy()

# @torch.no_grad()
def _get_slide_scores(net, test_loader, score_loader):
    # 1. read scores
    # 2. get A
    n = len(test_loader)
    scores = torch.zeros(n, device=device)
    net.eval()
    net = net.to(device)

    iterator = iter(score_loader)
    for b_idx, (data, target) in enumerate(test_loader):
        score = next(iterator)
        score = score.to(device)
        data, target = data.to(device), target.to(device)

        output, A = net(data)

        scores[b_idx] = torch.mm(A.transpose(0, 1), score)
    
    return scores



# load_mil, load, dataset (embed), histotypes, dataset_scores

def get_scores(args):
    net = KimiaNet()
    net.load_state_dict(preprocess_load(args.load, ngpu=1))

    train_set = OVEM(args.dataset, args.histotypes, debug=args.debug) # TODO
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    in_score, right_score, wrong_score = _get_patch_scores(net, train_loader, use_xent=True, in_dist=True) 

    net = VarMIL(dataset=args.dataset, histotypes=args.histotypes)
    net.load_state_dict(preprocess_load(args.load_mil, ngpu=1))
    if args.debug:
        print("Varmil loaded correctly?")
        pdb.set_trace()

    train_set = OVMIL(args.dataset, args.histotypes) # TODO 
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    score_set = OVSCORE(args.dataset_scores) # TODO
    score_loader = DataLoader(score_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    if args.debug:
        print("Datasets loaded correctly?")
        pdb.set_trace()

    slide_score = _get_slide_scores(net, train_loader, score_loader) 
    if args.debug:
        print("Scores calculated correctly?")
        pdb.set_trace()

    return in_score, right_score, wrong_score, slide_score

def get_ood_scores(args):
    net = KimiaNet()
    net.load_state_dict(preprocess_load(args.load, ngpu=1))

    ood_set = OVEM(args.dataset_ood, args.histotypes, debug=args.debug, ood=True) # TODO
    ood_loader = DataLoader(ood_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    ood_score = _get_patch_scores(net, ood_loader, use_xent=True)

    net = VarMIL(dataset=args.dataset, histotypes=args.histotypes)
    net.load_state_dict(preprocess_load(args.load_mil, ngpu=1))

    ood_set = OVMIL(args.dataset_ood, args.histotypes) # TODO 
    ood_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    score_set = OVSCORE(args.dataset_ood_scores) # TODO
    score_loader = DataLoader(score_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    slide_score = _get_slide_scores(net, ood_loader, score_loader) 

    return ood_score, slide_score

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1 # outliers are labeled as 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def show_performance(pos, neg, method_name='Ours', recall_level=0.95):
    '''
    :param pos: outlier or wrongly predicted example scores
    :param neg: inlier or correctly predicted scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))


# 1. find patch scores 
def main():
    print('Starting...')
    parser = argparse.ArgumentParser(description='Trains a VarMIL Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='', 
                        help='root_dir of image embeddings as .pt files. Path structure: root_dir/class_label/*.pt')
    parser.add_argument('--dataset_scores', type=str, default='', 
                        help='root_dir of image scores as .pt files. Path structure: root_dir/class_label/*.pt')
    parser.add_argument('--dataset_ood', type=str, default='')
    parser.add_argument('--dataset_ood_scores', type=str, default='')
    parser.add_argument('--histotypes', action='store', type=str, nargs="+", 
                        help='space separated str IDs specifying histotype labels')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--load_mil', type=str, default='')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--debug', action='store_true', default=False)

    print('Parsing arguments...')
    args = parser.parse_args()

    print('Getting scores...')
    in_score, right_score, wrong_score, slide_score = get_scores(args)
    out_score, out_slide_score = get_ood_scores(args)
    auroc, aupr, fpr = get_measures(out_slide_score, slide_score)
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    print('FPR95:\t\t\t{:.2f}'.format(100 * fpr))
    show_performance(out_slide_score, slide_score, method_name='VOS-VarMIL', recall_level=0.95)
    


if __name__ == '__main__':
    print("?")
    # pdb.set_trace()
    main()

