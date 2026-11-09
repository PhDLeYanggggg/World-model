"""Predict easy-case support separately from harm conditional on support."""

import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_av2_harm_quantile import QUANTILE


class EasyGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.edits = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(48+42, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 5))

    def forward(self, edits, context):
        z = self.edits(edits)
        return self.head(torch.cat((z.mean(1), z.amax(1), context), dim=1))


def loss(logits, y):
    support, safe, *_ = y.unbind(1)
    unconditional = F.binary_cross_entropy_with_logits(logits[:, :2], y[:, :2])
    conditional = F.binary_cross_entropy_with_logits(logits[:, 2], safe, reduction="none")
    error = y[:, 2:]-logits[:, 3:]
    pinball = torch.maximum(QUANTILE*error, (QUANTILE-1.)*error).mean(1)
    return unconditional+((conditional+4.*pinball)*support).sum()/support.sum().clamp_min(1)


def probabilities(logits):
    p = logits[..., :3].sigmoid()
    return torch.stack((p[..., 0], p[..., 1], p[..., 0]*p[..., 2], p[..., 2]), dim=-1)
