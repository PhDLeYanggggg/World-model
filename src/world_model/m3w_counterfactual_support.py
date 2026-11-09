"""Reference transitions for a single query replacement, not safety certificates."""
import numpy as np
import torch
from torch import nn


def loss_events(original_easy, current_easy, replacement_easy):
    """Return query-local and whole-record loss under a single substitution."""
    if not 0 <= current_easy <= original_easy:
        raise ValueError('Current easy support must be a subset of the original record')
    missing = np.asarray(replacement_easy) == 0
    local = (current_easy > 0) & missing
    whole = (original_easy > 0) & (original_easy == current_easy) & missing
    return np.stack((local, whole), axis=-1)


class Value(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def objective(net, x, y):
    z = net(x)
    gain = nn.functional.smooth_l1_loss(z[:, 0], y[:, 0], beta=.025)
    loss = nn.functional.binary_cross_entropy_with_logits(z[:, 1], y[:, 1])
    return gain + .2*loss
