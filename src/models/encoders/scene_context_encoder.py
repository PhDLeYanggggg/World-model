from __future__ import annotations

import torch
from torch import nn


class SceneContextEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

    def forward(self, features):
        return self.net(features)
