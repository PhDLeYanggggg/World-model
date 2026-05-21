from __future__ import annotations

import torch
from torch import nn


class HorizonDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.residual = nn.Linear(hidden_dim, 2)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, features):
        h = self.net(features)
        return self.residual(h), self.gate(h)
