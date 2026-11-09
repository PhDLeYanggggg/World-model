"""Factor the rare recording loss into local loss and uniqueness."""
import torch
from torch import nn


class Value(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


def objective(net, x, y):
    z = net(x)
    gain = nn.functional.smooth_l1_loss(z[:, 0], y[:, 0], beta=.025)
    local = nn.functional.binary_cross_entropy_with_logits(z[:, 1], y[:, 1])
    conditional = nn.functional.binary_cross_entropy_with_logits(z[:, 2], y[:, 2], reduction='none')
    conditional = (conditional*y[:, 1]).sum()/y[:, 1].sum().clamp_min(1.)
    return gain + .2*(local+conditional)


def risks(z):
    local = z[:, 1].sigmoid()
    return local, local*z[:, 2].sigmoid()
