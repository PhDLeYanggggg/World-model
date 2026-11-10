"""Estimate whether a full-recording pass survives background deletion."""

import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_exchange_risk_heads as risk


class StabilityRanker(risk.RiskRanker):
    def __init__(self, width):
        super().__init__(width)
        self.model[-1] = nn.Linear(32, 8)


def loss(scores, transition, gain, status, stable, auxiliary):
    base = risk.loss(scores[:, :7], transition, gain, status, True)
    if not auxiliary:
        return base
    keep = (status == 3) & (stable >= 0)
    logit = scores[:, 7]
    target = stable.clamp_min(0).to(logit.dtype)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    return base + (bce * keep).sum() / keep.sum().clamp_min(1)


def rank(scores, robust):
    p = risk.success_log_probability(scores[:, :7])
    return int((p + F.logsigmoid(scores[:, 7]) if robust else p).argmax())
