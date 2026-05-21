from __future__ import annotations

import torch
from torch import nn


class DynamicsDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
