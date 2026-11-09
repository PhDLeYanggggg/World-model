"""Expected gain, risk and evidence support for exact-slot trajectory edits."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_causal_risk_moments import CONTEXT, MomentHead, excess
from src.world_model.m3w_exact_slot_costs import CostHead


class SupportedCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = CostHead()
        self.moments = MomentHead()
        self.support = nn.Sequential(nn.Linear(CONTEXT, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, x, radius):
        return self.gain(x)[..., :2], self.moments(x, radius), self.support(x[..., :CONTEXT]).squeeze(-1)


def feasible(gains, moments, supports, easy, ledger, past_support, threshold):
    """An uncalibrated evidence gate, not a probability bound for a recording."""
    slack, denominators = excess(moments+ledger, easy)
    return ((gains.min(1) > 1e-10) & (denominators > 1e-10).all(1) & (slack <= 1e-10).all(1)
            & (np.maximum(supports, past_support) >= threshold))


def packed(values, options):
    return np.where((options >= 0)[..., None], values[np.maximum(options, 0)], 0.)
