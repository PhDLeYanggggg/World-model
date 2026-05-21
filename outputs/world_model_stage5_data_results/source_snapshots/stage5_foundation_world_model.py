from __future__ import annotations

import torch
from torch import nn

from src.models.decoders.dynamics_decoder import DynamicsDecoder
from src.models.decoders.latent_future_decoder import LatentFutureDecoder
from src.models.encoders.dataset_domain_encoder import DatasetDomainEncoder
from src.models.encoders.entity_encoder import EntityEncoder
from src.models.encoders.interaction_graph_encoder import InteractionGraphEncoder


class Stage5FoundationWorldModel(nn.Module):
    """Deterministic-first 2.5D world model scaffold.

    Latent/stochastic mode is disabled by default and should only be enabled
    after deterministic gates pass.
    """

    def __init__(self, entity_dim: int, hidden_dim: int = 128, num_domains: int = 32, latent_enabled: bool = False):
        super().__init__()
        self.latent_enabled = latent_enabled
        self.entity_encoder = EntityEncoder(entity_dim, hidden_dim)
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.interaction_encoder = InteractionGraphEncoder(hidden_dim)
        self.domain_encoder = DatasetDomainEncoder(num_domains, hidden_dim)
        self.latent_decoder = LatentFutureDecoder(hidden_dim)
        self.dynamics_decoder = DynamicsDecoder(hidden_dim, output_dim=4)
        self.uncertainty_head = nn.Linear(hidden_dim, 3)

    def forward(self, past: torch.Tensor, domain_ids: torch.Tensor | None = None) -> dict:
        # past: [B, T, N, F]
        b, t, n, f = past.shape
        x = self.entity_encoder(past.reshape(b * t * n, f)).reshape(b * n, t, -1)
        temporal, _ = self.temporal_encoder(x)
        current = temporal[:, -1].reshape(b, n, -1)
        if domain_ids is not None:
            current = current + self.domain_encoder(domain_ids).unsqueeze(1)
        current = self.interaction_encoder(current)
        latent, mu, logvar = self.latent_decoder(current, enabled=self.latent_enabled)
        residual = self.dynamics_decoder(latent)
        uncertainty = self.uncertainty_head(latent)
        return {"residual": residual, "uncertainty": uncertainty, "latent_mu": mu, "latent_logvar": logvar}
