import os
import numpy as np
import json
import re
import glob
import pdb
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.kimianet_virtual import KimiaNet
from models.wrn50_virtual import WideResNet50_2
from mil.varmil import VarMIL
from PIL import Image


def dict_to_device(orig, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    new = {}
    for k,v in orig.items():
        new[k] = v.to(device)
    return new

def class_proportion(split, histotypes, chunk=0):
    with open(split) as f:
        data = json.load(f)
        training_data = data['chunks'][chunk]['imgs']
        
    class_prop = torch.zeros(len(histotypes))
    for i, h in enumerate(histotypes):
        r = re.compile(f".*Tumor/{h}/.*")
        x = list(filter(r.match, training_data))
        class_prop[i] = len(x) / len(training_data)

    return class_prop

def preprocess_load(state_dict_path, ngpu):
    state_dict = torch.load(state_dict_path)
    new_state_dict = {}

    if ngpu > 1 and list(state_dict.keys())[0].startswith('module.'): # n to n
        return state_dict
    elif ngpu == 1 and list(state_dict.keys())[0].startswith('module.'): # n to 1
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
        return new_state_dict
    elif ngpu > 1: # 1 to n
        for key, value in state_dict.items():    
            new_key = 'module.' + key
            new_state_dict[new_key] = value
        return new_state_dict
    else: # 1 to 1
        return state_dict

def _pad_array_with_zeros(array, target_rows):
    array = np.array(array)
    current_rows = array.shape[0]
    padding_rows = target_rows - current_rows
    
    if padding_rows <= 0:
        return array
    
    pad_width = [(0, padding_rows)] + [(0, 0)] * (array.ndim - 1)
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
    
    return padded_array

def load_scores(folder_path):
    '''
    Gets patch scores if they have already been computed and saved as .npy files.
    '''
    concatenated_vectors = [np.array([]), np.array([]), np.array([])]

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                arr = np.load(file_path)

                for i in range(3):
                    concatenated_vectors[i] = np.concatenate((concatenated_vectors[i], arr[:, i]))

    return concatenated_vectors

def _get_patch_scores(output, log_reg, energy_weights, T=1.0):
    '''
    Computes xent, confidence calibrated softmax, and MLP scores learned during training. Return as numpy arrays. 
    output (torch.tensor): output of the model
    log_reg (torch.nn.Module): log_reg layer
    energy_weights (torch.nn.Module): energy_weights layer
    '''
    _to_np = lambda x: x.data.cpu().numpy()

    score = torch.zeros((3, 1), device=output.get_device())
    score[0] = output.mean(1) - torch.logsumexp(output, dim=1, keepdim=False)
    score[1] = -(T * torch.logsumexp(output / T, dim=1))
    # energy score used during training
    m, _ = torch.max(output, dim=1, keepdim=True)
    value0 = output - m
    score[2] = m + torch.log(torch.sum(
               F.relu(energy_weights.weight) * torch.exp(value0), dim=1, keepdim=False))
    out = log_reg(score[2])

    return _to_np(score.transpose(0, 1))[0], _to_np(out)
    
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def generate_embeds_and_scores(root_dir, model_weights_path, output_dir, ngpu=1, debug=False, save=False, T=1.0):
    '''
    Generates embeddings and scores for all patches in the dataset. Optionally saves embeddings. 
    root_dir (str): path to the patch dataset 
    model_weights_path (str): path to the model weights of net, weight_energy, and logistic_regression
    output_dir (str): path to save the embeddings

    Return: 
    scores (numpy.ndarray list): (num_slides, num_patches, 3) array of xent, confidence calibrated softmax, and MLP scores for each class
    bin_cls (numpy.ndarray list): (num_slides, num_patches, 2) array of binary classification scores (0: ood, 1: id) for each class
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pre-trained models and set to evaluation mode
    model = KimiaNet(num_classes=5, freeze=False, drop_rate=0.0)
    model.load_state_dict(preprocess_load(os.path.join(model_weights_path, 'net.pt'), ngpu))
    model.to(device)
    model.eval()
    energy_weights = torch.nn.Linear(5, 1)
    energy_weights.load_state_dict(preprocess_load(os.path.join(model_weights_path, 'weight_energy.pt'), ngpu))
    energy_weights.to(device)
    log_reg = torch.nn.Linear(1, 2)
    log_reg.load_state_dict(preprocess_load(os.path.join(model_weights_path, 'logistic_regression.pt'), ngpu))
    log_reg.to(device)

    # Define image preprocessing 
    mean = [x / 255 for x in [207.79, 177.028, 209.72]]
    std = [x / 255 for x in [28.79, 33.26, 18.41]]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    scores = []
    bin_clses = []
    # Iterate through all class labels, image labels, and patch images
    # tmp = [d for d in os.listdir(root_dir) if d != 'EC' and d != 'MC' and os.path.isdir(os.path.join(root_dir, d))]
    # for class_label in tmp:
    for class_label in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_label)
        num_img = len(os.listdir(class_path))
        scores_cls = np.zeros((num_img, 150, 3))
        bin_cls = np.zeros((num_img, 150, 2))

        for img_idx, image_label in enumerate(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_label, "512", "20")
            patches = glob.glob(f"{image_path}/*.png")
            patches = sorted(patches, key=natural_sort_key)

            embeddings = torch.zeros((150, 1024), device=device)
            scores_img = np.zeros((150, 3))
            bin_cls_img = np.zeros((150, 2))

            for idx, patch_path in enumerate(patches[:150]):
                img = Image.open(patch_path)
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out, emb = model(img_tensor)
                    sc, cls = _get_patch_scores(out, log_reg, energy_weights, T=T)
                embeddings[idx] = emb.squeeze().cpu()
                scores_img[idx] = sc
                bin_cls_img[idx] = cls

            scores_cls[img_idx] = scores_img
            bin_cls[img_idx] = bin_cls_img

            # Save embeddings, scores, and binary classification scores by class
            # save scores in memory for easier slide prediction aggregation
            if save:
                os.makedirs(os.path.join(output_dir,'embeds', class_label), exist_ok=True)
                os.makedirs(os.path.join(output_dir,'scores', class_label), exist_ok=True)
                os.makedirs(os.path.join(output_dir,'log_cls', class_label), exist_ok=True)
                torch.save(embeddings, os.path.join(output_dir, 'embeds', class_label, f"{image_label}.pt"))
                np.save(os.path.join(output_dir, 'scores', class_label, f"{image_label}.npy"), scores_img)
                np.save(os.path.join(output_dir, 'log_cls', class_label, f"{image_label}.npy"), bin_cls_img)
        scores.append(scores_cls)
        bin_clses.append(bin_cls)

    return scores, bin_clses

def generate_attentions_and_scores(root_dir, model_weights_path, output_dir, save=False, ngpu=1, debug=False, use_xent=False, score='energy', T=1.0):
    '''
    Generates attention maps and scores for all slides in the dataset.
    root_dir (str): path to the emb, scores dataset. Requires: root_dir/(embeds|log_cls|scores)/class_label/image_label.(pt|npy)
    model_weights_path (str): path to the model weights of att_net
    output_dir (str): path to save the attentions

    Returns aggregated scores of patches
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _to_np = lambda x: x.data.cpu().numpy()

    # Load 
    model = VarMIL(num_classes=5)
    model.load_state_dict(preprocess_load(model_weights_path, ngpu))
    model.to(device)
    model.eval()
    sld_scos_all = []
    log_cls_all = []

    for class_label in os.listdir(os.path.join(root_dir, 'embeds')):
        class_path = os.path.join(root_dir, 'embeds', class_label)
        sld_scos = [] # no.slide x 3
        log_cls = [] # no.slide x 2

        for emb_label in os.listdir(class_path):
            emb_label = str.split(emb_label, '.')[0]
            emb_path = os.path.join(class_path, f"{emb_label}.pt")
            emb = torch.load(emb_path)
            psco_path = os.path.join(root_dir, 'scores', class_label, f"{emb_label}.npy")
            ptch_sco = torch.from_numpy(np.load(psco_path))
            bn_cls_path = os.path.join(root_dir, 'log_cls', class_label, f"{emb_label}.npy")
            bn_cls = torch.from_numpy(np.load(bn_cls_path))
            
            emb = emb.unsqueeze(0).to(device)
            ptch_sco = ptch_sco.to(device)
            bn_cls = bn_cls.to(device)
            with torch.no_grad():
                _, att = model(emb)
                if debug:
                    print("hello")
                    pdb.set_trace()
                    sld_co=torch.mul(att.squeeze(), ptch_sco).sum(0)
                    print(torch.mul(att.squeeze(), ptch_sco)) # 150 x 3 
                    print(sld_co) # 1 x 3
                sld_sco = torch.mul(att.squeeze(), ptch_sco.transpose(0, 1)).sum(1)
                bn_cls_ = torch.mul(att.squeeze(), bn_cls.transpose(0, 1)).sum(1) 
            sld_scos.append(_to_np(sld_sco))
            log_cls.append(_to_np(bn_cls_))

            if save:
                os.makedirs(os.path.join(output_dir, 'att', class_label), exist_ok=True)
                torch.save(att.squeeze(), os.path.join(output_dir, 'att', class_label, f"{emb_label}.pt"))
        sld_scos = np.vstack(sld_scos)
        sld_scos_all.append(sld_scos) # 5 x no.slide x 3
        log_cls = np.vstack(log_cls)
        log_cls_all.append(log_cls) # 5 x no.slide x 2

    return sld_scos_all, log_cls_all


def att_training_split(input_json, output_json, emb_dir=None):
    '''
    Create training and validation sets for the attention model given patch-models training set split
    '''
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Get the image label and class label from the patch path
    _get_image_label = lambda path: {'slide_label': path.split('/')[-4], 'class_label': path.split('/')[-5]}

    slide_chunks = []
    for chunk in data['chunks']:
        slide_labels = set()
        for ptch_path in chunk['imgs']:
            lab = _get_image_label(ptch_path)
            label = lab['slide_label']
            if emb_dir is not None:
                label = os.path.join(emb_dir, lab['class_label'], f"{lab['slide_label']}.pt")
            slide_labels.add(label)

        slide_chunk = {
            'id': chunk['id'],
            'imgs': list(slide_labels)
        }
        slide_chunks.append(slide_chunk)

    slide_data = {
        'chunks': slide_chunks
    }

    with open(output_json, 'w') as f:
        json.dump(slide_data, f, indent=4)
