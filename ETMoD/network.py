import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Independent
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import init_func as init

def set_all_seeds(seed):
#   random.seed(seed)
#   os.environ('PYTHONHASHSEED') = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    """MLP"""
    # def __init__(self, n_inputs, n_outputs, hparams):
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        
        # self.num_layers = hparams['mlp_depth']
        self.num_layers = 3

        if self.num_layers > 1:
            self.input = nn.Linear(n_inputs, 64)
            # self.dropout = nn.Dropout(0)
            self.hiddens = nn.ModuleList([
                nn.Linear(64, 64)
                for _ in range(3-2)])
            self.output = nn.Linear(64, n_outputs)
        else:
            self.input = nn.Linear(n_inputs, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        if self.num_layers > 1:
            # x = self.dropout(x)
            x = F.relu(x)
            for hidden in self.hiddens:
                x = hidden(x)
                # x = self.dropout(x)
                x = F.relu(x)
            x = self.output(x)
        return x
    
class SDE(torch.nn.Module):
    noise_type = "diagonal"
    def __init__(self, n_inputs, n_outputs, device):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"  # 'ito':"euler","milstein","srk" 'stratonovich':"midpoint","milstein","reversible_heun"
        self.brownian_size = n_outputs # hparams["brownian_size"] # n_outputs // 2 if n_outputs > 16 else n_outputs  # 8
   
        self.mu1 = MLP(n_inputs, n_outputs)
        self.mu2 = MLP(n_inputs, n_outputs)

        self.sigma1 = MLP(n_inputs, n_outputs)
        self.sigma2 = MLP(n_inputs, n_outputs)
        self.state_size = n_inputs
        self.to(device)

    # Drift
    def f(self, t, x):
        
        self.device = "cuda" 

        t = t.view(-1, 1).expand(x.size(0), x.size(1)).to(self.device)

        x = self.mu1(x) + self.mu2(t)
        # print(self.mu1(x))
        return x 

    # Diffusion
    def g(self, t, x):
        self.device = "cuda" 

        t = t.view(-1, 1).expand(x.size(0), x.size(1)).to(self.device)

        sigma = F.softplus(self.sigma1(x)) + F.softplus(self.sigma2(t))

        return sigma  
