import torch
import wandb
import os
from ovca import OVCA
import json
import re

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

def class_proportion(split, histotypes):
    with open(split) as f:
        data = json.load(f)
        training_data = data['chunks'][0]['imgs']
        
    class_prop = torch.zeros(len(histotypes))
    for i, h in enumerate(histotypes):
        r = re.compile(f".*Tumor/{h}/.*")
        x = list(filter(r.match, training_data))
        class_prop[i] = len(x) / len(training_data)

    return class_prop

def preprocess_load(state_dict_path, ngpu):
    state_dict = torch.load(state_dict_path)
    new_state_dict = {}

    if ngpu > 1:
        return state_dict
    if ngpu == 1: # n to 1
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
    # else: # 1 to n
    #     for key, value in state_dict.items():    
    #         new_key = 'module.' + key
    #         new_state_dict[new_key] = value
    return new_state_dict
    

