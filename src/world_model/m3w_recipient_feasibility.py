"""Forecast attainable recipient-level risk, rather than each path's mean cost."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def attainable(gain, known, radius):
    """Best observed ADE improvement or smallest missing-label debit."""
    legal = radius > 0.
    best = np.where(legal, gain, -np.inf).max(1)
    debit = np.where(legal, radius, np.inf).min(1)
    use = legal.any(1)
    return np.where(use & known, best, 0.), np.where(use & ~known, debit, 0.), use


class RecipientHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 32),
                                     nn.SiLU(), nn.Linear(32, 8))

    def forward(self, x, bound):
        z = self.network(x)
        all_ref = F.softplus(z[:, 0])
        ref = torch.stack((all_ref, all_ref*z[:, 1].sigmoid()), dim=1)
        harm = bound*z[:, 3].sigmoid()
        cost = torch.stack((bound*z[:, 2].tanh(), harm, harm*z[:, 4].sigmoid()), dim=1)
        return ref, cost, z[:, 5:]


def loss(pred, base, gain, debit, known, easy, aux):
    ref, cost, logits = pred
    harm = (-gain).clamp_min(0.)
    refs = torch.stack((base, easy*base), 1)
    moments = torch.stack((gain-debit, harm+debit, easy*harm+debit), 1)
    out = F.mse_loss(ref, refs)+F.mse_loss(cost, moments)
    if aux:
        # Separate observable support, a locally safe easy anchor, and zero harm.
        tags = torch.stack((known, easy & (harm <= .02*base), known & (gain >= 0.)), 1)
        out = out+.25*F.binary_cross_entropy_with_logits(logits, tags.to(logits.dtype))
    return out


def nominate(cost, legal):
    """Common frozen choice of one path for each person, before any ranking."""
    gain, all_harm, easy_harm = cost.transpose(2, 0, 1)
    return np.where(legal, gain-4.*(all_harm+easy_harm), -np.inf).argmax(1)
