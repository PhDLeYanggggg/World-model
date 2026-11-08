"""Learn joint versus row-local supervision at an identical causal action budget."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def network(width):
    return nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 4))


def local_teacher(gain, incumbent):
    best = gain.argmax(1)
    return np.where(gain.max(1) > 0., best, incumbent)


def loss(net, x, target, incumbent):
    p = net(x)
    effects = p[..., :3]-p[:, incumbent, None, :3]
    preferences = p[..., 3]
    classification = F.cross_entropy(preferences, target[..., 3], label_smoothing=.05)
    return F.mse_loss(effects[..., 0], target[..., 0]) + .5*F.mse_loss(
        effects[..., 1:], target[..., 1:3]) + .25*classification


def choose(raw, incumbent, guarded):
    centered = raw[..., :3]-raw[:, incumbent, None, :3]
    allowed = centered[..., 0] > 0.
    if guarded:
        allowed &= (centered[..., 1:] <= 0.).all(2)
    allowed[:, incumbent] = True
    return np.argmax(np.where(allowed, raw[..., 3], -np.inf), axis=1)
