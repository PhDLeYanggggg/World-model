from __future__ import annotations

import torch
from torch import nn


class DatasetDomainEncoder(nn.Module):
    def __init__(self, num_domains: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_domains, hidden_dim)

    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(domain_ids)
