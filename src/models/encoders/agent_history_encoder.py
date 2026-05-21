from __future__ import annotations

import torch
from torch import nn


class AgentHistoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 48):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, history):
        _, hidden = self.gru(history)
        return hidden[-1]
