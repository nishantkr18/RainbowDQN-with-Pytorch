import os
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output

class CategoricalDQN(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        hid_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(CategoricalDQN, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid_dim), 
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), 
            nn.ReLU(), 
            nn.Linear(hid_dim, out_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        # print(self.support.shape)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
#         print(q_atoms.shape)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist