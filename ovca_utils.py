import torch
import wandb
import os
from ovca import OVCA

def dict_to_device(orig, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    new = {}
    for k,v in orig.items():
        new[k] = v.to(device)
    return new

def init_wb(name):
    wandb.login()
    wandb.init(project="a2", name=name)
    wandb.define_metric("train_loss", step_metric="iteration")
    wandb.define_metric("val_loss",step_metric="epoch")

def cross_val(args):
    root = args.splits_dir # '/projects/ovcare/classification/cshi/OoD/data/512_20/create_splits/splits'
    splits = os.listdir(root)
    split_paths = [os.path.join(root, x) for x in splits]

    for i, split in enumerate(split_paths):
        train_set = OVCA(split, 0, transform=None, target_transform=None, idx_to_label=idx_to_label, key_word='Tumor')
        val_set = OVCA(split, 1, transform=None, target_transform=None, idx_to_label=idx_to_label, key_word='Tumor') 
        test_set = OVCA(split, 2, transform=None, target_transform=None, idx_to_label=idx_to_label, key_word='Tumor') 

