"""Continuous gain and harm supervision for frozen nested forecast actions."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PrefixValue(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 10))

    def forward(self, x):
        return self.net(x).reshape(-1, 2, 5)


def objective(prediction, target, valid, *, risk):
    n = 5 if risk else 2
    return F.smooth_l1_loss(prediction[valid, :n], target[valid, :n], beta=.025)


def decision_scores(prediction, scale, legal, *, penalty):
    """Convert incremental utility to native sums; keep fallback exactly zero."""
    if (prediction.shape != (len(scale), 2, 5) or legal.shape != (len(scale), 3)
            or not np.isfinite(prediction).all() or not np.isfinite(scale).all() or (scale <= 0.).any()):
        raise ValueError('Invalid past-only prefix values or scale')
    score = prediction[..., :2].min(-1)
    if penalty:
        score = score - prediction[..., 2:].max(-1).clip(0.)
    result = np.column_stack((np.zeros(len(scale)), score.astype(float) * scale[:, None]))
    return np.where(legal, result, -np.inf)
