from __future__ import annotations

import torch
from torch import nn


class NeighborInteractionEncoder(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, features):
        return self.net(features)
