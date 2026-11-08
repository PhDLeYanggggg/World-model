"""Matched-budget preference learning for current-query interventions."""

import numpy as np
from torch import nn
from torch.nn import functional as F

GEOMETRY = 4


def risk_label(gain, excess, easy, mass, ref_easy, ols_offset):
    return bool(easy > 0. and mass > 0. and ref_easy > 0.
                and (excess <= [0., 0., .02*ref_easy]).all()
                and gain[0] > 0. and gain[1]+ols_offset > 0.)


def choose(z, guarded=False):
    if z.ndim != 2 or z.shape[1] != 6 or not np.isfinite(z).all():
        raise ValueError("Expected finite matched-slate scores")
    lower = z[:, 0]-z[0, 0]-np.maximum(z[:, 1], 0.)
    lower[0] = 0.
    if guarded:
        allowed = ((z[:, 2:5] <= z[0, 2:5]).all(1)) & (z[:, 5] >= z[0, 5])
        lower = np.where(allowed, lower, -np.inf)
    return int(lower.argmax())


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32),
                         nn.SiLU(), nn.Linear(32, 6))


def loss(net, x, y, ranking):
    z = net(x)
    point = F.smooth_l1_loss(z[:, :5], y[:, :5], beta=.05)
    binary = F.binary_cross_entropy_with_logits(z[:, 5], y[:, 5])
    if not ranking:
        return point+.1*binary
    # An interval overlapping zero supplies no hard preference label.
    certainty = (y[:, 0].abs()-y[:, 1]).clamp(0., 1.)
    rank = F.binary_cross_entropy_with_logits(z[:, 0]/.1, (y[:, 0] > 0.).to(z.dtype), reduction="none")
    return point+.1*binary+.25*(certainty*rank).mean()
