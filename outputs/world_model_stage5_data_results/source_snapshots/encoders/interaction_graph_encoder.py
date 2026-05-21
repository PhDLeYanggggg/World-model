from __future__ import annotations

import torch
from torch import nn


class InteractionGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + y)
