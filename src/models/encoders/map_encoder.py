from __future__ import annotations

import torch
from torch import nn


class MapEncoder(nn.Module):
    def __init__(self, map_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(map_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, map_features: torch.Tensor | None, fallback_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        if map_features is None:
            return torch.zeros((*fallback_shape, self.net[-1].out_features), device=device)
        return self.net(map_features)
