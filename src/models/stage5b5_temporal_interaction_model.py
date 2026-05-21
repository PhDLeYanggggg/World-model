from __future__ import annotations

import torch
from torch import nn

from src.models.encoders.agent_history_encoder import AgentHistoryEncoder
from src.models.encoders.neighbor_interaction_encoder import NeighborInteractionEncoder
from src.models.encoders.scene_context_encoder import SceneContextEncoder
from src.models.decoders.horizon_decoder import HorizonDecoder


class Stage5B5TemporalInteractionModel(nn.Module):
    def __init__(self, history_dim: int = 9, num_datasets: int = 8, hidden_dim: int = 48, residual_clip: float = 5.0):
        super().__init__()
        self.history_encoder = AgentHistoryEncoder(history_dim, hidden_dim)
        self.neighbor_encoder = NeighborInteractionEncoder(4, 24)
        self.scene_encoder = SceneContextEncoder(3, 16)
        self.dataset_embedding = nn.Embedding(num_datasets, 8)
        self.decoder = HorizonDecoder(hidden_dim + 24 + 16 + 8 + 1, 64)
        self.residual_clip = residual_clip

    def forward(self, history, neighbor_features, scene_features, dataset_id, horizon_frac):
        h = self.history_encoder(history)
        n = self.neighbor_encoder(neighbor_features)
        s = self.scene_encoder(scene_features)
        d = self.dataset_embedding(dataset_id)
        x = torch.cat([h, n, s, d, horizon_frac[:, None]], dim=-1)
        residual, gate = self.decoder(x)
        residual = torch.tanh(residual / self.residual_clip) * self.residual_clip
        return residual * gate, gate
