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
from q_networks import vanilla_dqn

class VanillaDQN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        """Initializes the network"""
        super(VanillaDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid_dim), 
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), 
            nn.ReLU(), 
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)