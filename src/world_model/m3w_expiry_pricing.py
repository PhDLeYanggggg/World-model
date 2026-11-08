"""Separate the value of a moved edit from withdrawal of the original edit."""

import numpy as np
from scipy.special import expit
from torch import nn

ARMS = ("direct", "mixture", "priced", "geometry")


class Value(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 7))

    def forward(self, x):
        return self.net(x)


def loss(net, x, y):
    z = net(x)
    events = nn.functional.binary_cross_entropy_with_logits(z[:, :3], y[:, :3])
    err = (z[:, 3:]-y[:, [3, 3, 3, 4]]).square()
    move = y[:, 0]
    conditional = (err[:, 1]*move).sum()/move.sum().clamp_min(1)
    expiry = 1.-move
    withdrawal = (err[:, 2]*expiry).sum()/expiry.sum().clamp_min(1)
    return events + err[:, 0].mean() + .5*(conditional+withdrawal+err[:, 3].mean())


def score(z, distance):
    """Geometry bounds withdrawal only, not the full predicted decision."""
    p = expit(z[:, 0])
    move = p*z[:, 4]
    expiry = 1.-p
    return dict(direct=z[:, 3], mixture=move+expiry*z[:, 5],
                priced=move+expiry*np.minimum(np.minimum(z[:, 5], z[:, 6]), 0.),
                geometry=move-expiry*distance)
