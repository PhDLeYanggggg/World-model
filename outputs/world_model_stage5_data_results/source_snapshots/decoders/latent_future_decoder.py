from __future__ import annotations

import torch
from torch import nn


class LatentFutureDecoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int = 16):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.out = nn.Linear(hidden_dim + latent_dim, hidden_dim)

    def forward(self, x: torch.Tensor, enabled: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        logvar = self.logvar(x)
        if enabled:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = torch.zeros_like(mu)
        return self.out(torch.cat([x, z], dim=-1)), mu, logvar
