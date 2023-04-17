import torch
import torch.nn as nn
import torch.nn.functional as F

class VarMIL(nn.Module):
    def __init__(self, in_size=1024, dim=128, num_classes=5):
        super().__init__()
        dim = 128 
        torch.autograd.set_detect_anomaly(True)
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention  = nn.Sequential(nn.Linear(in_size, dim),
                                       nn.Tanh(),
                                       nn.Linear(dim, 1))
        self.classifier = nn.Sequential(nn.Linear(2*in_size, dim),
                                       nn.ReLU(),
                                       nn.Linear(dim, num_classes))

    def forward(self, x):
        """
        x   (input)            : B (batch size) x K (nb_patch) x out_channel
        A   (attention weights): B (batch size) x K (nb_patch) x 1
        M   (weighted mean)    : B (batch size) x out_channel
        S   (std)              : B (batch size) x K (nb_patch) x out_channel
        V   (weighted variance): B (batch size) x out_channel
        nb_patch (nb of patch) : B (batch size)
        M_V (concate M and V)  : B (batch size) x 2*out_channel
        out (final output)     : B (batch size) x num_classes
        """
        b, k, c = x.shape
        A = self.attention(x)
        A = A.masked_fill((x == 0).all(dim=2).reshape(A.shape), -9e15) # filter padded rows
        A = F.softmax(A, dim=1)                                        # softmax over K
        M = torch.einsum('b k d, b k o -> b o', A, x)                  # d = 1 because A is B x K x 1
        S = torch.pow(x-M.reshape(b,1,c), 2)
        V = torch.einsum('b k d, b k o -> b o', A, S)
        nb_patch = (torch.tensor(k).expand(b)).to(self.device)
        nb_patch = nb_patch - torch.sum((x == 0).all(dim=2), dim=1)    # filter padded rows
        nb_patch = nb_patch / (nb_patch - 1)                           # I / I-1 to make cases with 1 patch inf 
        nb_patch = torch.nan_to_num(nb_patch, posinf=1)                # for cases when we have only 1 patch (inf)
        V = V * nb_patch[:, None]                                      # broadcasting
        M_V = torch.cat((M, V), dim=1)
        out = self.classifier(M_V)
        return out, A


class DeepMIL(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        dim = 128
        torch.autograd.set_detect_anomaly(True)
        self.attention  = nn.Sequential(nn.Linear(in_size, dim),
                                       nn.Tanh(),
                                       nn.Linear(dim, 1))
        self.classifier = nn.Sequential(nn.Linear(in_size, dim),
                                       nn.ReLU(),
                                       nn.Linear(dim, num_classes))

    def forward(self, x):
        """
        x   (input)            : B (batch size) x K (nb_patch) x out_channel
        A   (attention weights): B (batch size) x K (nb_patch) x 1
        M   (weighted mean)    : B (batch size) x out_channel
        out (final output)     : B (batch size) x num_classes
        """
        A = self.attention(x)
        A = A.masked_fill((x == 0).all(dim=2).reshape(A.shape), -9e15) # filter padded rows
        A = F.softmax(A, dim=1)   # softmax over K
        M = torch.einsum('b k d, b k o -> b o', A, x) # d is 1 here
        out = self.classifier(M)
        return A, out