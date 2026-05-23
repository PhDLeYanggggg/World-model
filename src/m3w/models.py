from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from src.m3w.token_schema import TOKEN_NAMES, TokenSchema


class M3WTokenizer(nn.Module):
    def __init__(self, schema: TokenSchema, token_dim: int) -> None:
        super().__init__()
        self.schema = schema
        self.token_dim = token_dim
        self.projections = nn.ModuleDict()
        for token in TOKEN_NAMES:
            in_dim = len(schema.token_to_features[token])
            self.projections[token] = nn.Sequential(nn.Linear(in_dim, token_dim), nn.LayerNorm(token_dim), nn.GELU())
        self.token_type = nn.Parameter(torch.randn(len(TOKEN_NAMES), token_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        for i, token in enumerate(TOKEN_NAMES):
            idx = torch.tensor(self.schema.token_to_features[token], device=x.device, dtype=torch.long)
            tok = self.projections[token](x.index_select(dim=1, index=idx)) + self.token_type[i]
            tokens.append(tok)
        return torch.stack(tokens, dim=1)


class JEPAEncoder(nn.Module):
    def __init__(self, token_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens).mean(dim=1)


class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class TransformerDynamics(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.out(self.encoder(tokens).mean(dim=1))


class M3WHeads(nn.Module):
    def __init__(self, hidden_dim: int, num_baselines: int) -> None:
        super().__init__()
        self.fde = nn.Linear(hidden_dim, num_baselines)
        self.failure = nn.Linear(hidden_dim, 1)
        self.goal = nn.Linear(hidden_dim, 4)
        self.interaction = nn.Linear(hidden_dim, 1)
        self.occupancy = nn.Linear(hidden_dim, 1)
        self.validity = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "log_fde": self.fde(h),
            "failure_logit": self.failure(h).squeeze(-1),
            "goal_logits": self.goal(h),
            "interaction_logit": self.interaction(h).squeeze(-1),
            "occupancy": torch.sigmoid(self.occupancy(h).squeeze(-1)),
            "validity_logit": self.validity(h).squeeze(-1),
        }


class M3WModel(nn.Module):
    def __init__(self, schema: TokenSchema, config: Dict, variant: str) -> None:
        super().__init__()
        self.variant = variant
        token_dim = int(config["token_dim"])
        latent_dim = int(config["latent_dim"])
        hidden_dim = int(config["hidden_dim"])
        self.tokenizer = M3WTokenizer(schema, token_dim)
        self.jepa_encoder = JEPAEncoder(token_dim, latent_dim)
        self.jepa_predictor = JEPAPredictor(latent_dim)
        self.transformer = TransformerDynamics(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            layers=int(config["transformer_layers"]),
            heads=int(config["transformer_heads"]),
            dropout=float(config.get("dropout", 0.1)),
        )
        self.jepa_to_hidden = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.GELU())
        self.hybrid_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.heads = M3WHeads(hidden_dim, num_baselines=7)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenizer(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens(x)
        if self.variant == "jepa_only":
            return self.jepa_to_hidden(self.jepa_encoder(tokens))
        if self.variant == "transformer_only":
            return self.transformer(tokens)
        j = self.jepa_to_hidden(self.jepa_encoder(tokens))
        t = self.transformer(tokens)
        return self.hybrid_fusion(torch.cat([j, t], dim=-1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.heads(self.encode(x))

    def jepa_forward(self, x: torch.Tensor, mask_ratio: float) -> Dict[str, torch.Tensor]:
        tokens = self.tokens(x)
        target = self.jepa_encoder(tokens).detach()
        mask = torch.rand(tokens.shape[:2], device=tokens.device) < mask_ratio
        masked = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
        pred = self.jepa_predictor(self.jepa_encoder(masked))
        return {"pred": pred, "target": target}
