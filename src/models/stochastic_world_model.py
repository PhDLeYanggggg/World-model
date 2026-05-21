from __future__ import annotations

import torch

from src.models.neural_residual_world_model import DeterministicResidualWorldModel


class StochasticResidualWorldModel(DeterministicResidualWorldModel):
    def __init__(self, entity_dim: int, neighbor_dim: int, obstacle_dim: int, latent_dim: int = 4) -> None:
        super().__init__(entity_dim, neighbor_dim, obstacle_dim, latent_dim=latent_dim)

    def sample_latent(self, batch: int, scale: float = 0.7) -> torch.Tensor:
        return torch.randn((batch, self.latent_dim), dtype=torch.float32) * scale
