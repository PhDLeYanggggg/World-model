"""Decide whether to defer one gain edit without changing the support edit."""

from itertools import islice

import numpy as np
from torch import nn
from torch.nn import functional as F

DEADLINE = 8


def first_offer(packets, radius):
    """Consume only observed queries, stopping on the first legal radius match."""
    if not np.isfinite(radius) or radius <= 0.:
        raise ValueError("Expected a positive finite gain radius")
    for elapsed, (support_seen, rows, size, gain) in enumerate(islice(packets, DEADLINE), 1):
        if not support_seen:
            continue
        legal = size >= radius
        if legal.any():
            return elapsed, int(rows[np.where(legal, gain, -np.inf).argmax()])
    return None


def choose_wait(z, guard):
    if z.ndim != 2 or z.shape[1] != 3 or not np.isfinite(z).all():
        raise ValueError("Invalid deferral scores")
    positive = z[:, 0] > 0.
    if guard:
        positive &= (z[:, 1:] < 0.).all(1)
    return positive


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 3))


def loss(net, x, y):
    z = net(x)
    return (F.smooth_l1_loss(z[:, 0], y[:, 0], beta=.05)
            + .15 * F.binary_cross_entropy_with_logits(z[:, 1:], y[:, 1:]))
