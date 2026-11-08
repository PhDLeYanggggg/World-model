"""Predict baseline-relative cost and make count-matched current-query choices."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def choose(gain, legal, cap):
    """Return one family per selected agent, maximizing additive gain."""
    if not np.isfinite(gain).all() or gain.shape != legal.shape:
        raise ValueError("Invalid current-query action matrix")
    score = np.where(legal, gain, -np.inf)
    family = score.argmax(1)
    value = score[np.arange(len(score)), family]
    chosen = np.flatnonzero(np.isfinite(value))
    chosen = chosen[np.argsort(-value[chosen], kind="stable")[:cap]]
    out = np.full(len(score), -1, dtype=int)
    out[chosen] = family[chosen]
    return out


def network(width):
    return nn.Sequential(nn.Linear(width, 40), nn.Tanh(), nn.Linear(40, 24),
                         nn.Tanh(), nn.Linear(24, 3))


def loss(net, x, y):
    return F.smooth_l1_loss(net(x), y, beta=.1)
