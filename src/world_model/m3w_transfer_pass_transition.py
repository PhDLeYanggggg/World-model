"""Learn a ternary recording-pass transition alongside paired utility."""
import numpy as np
import torch
from torch import nn


def transition(before, after):
    """0 unchanged, 1 newly qualified, 2 destroyed."""
    a, b = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    return np.where(~a & b, 1, np.where(a & ~b, 2, 0))


class Value(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,64), nn.SiLU(), nn.Linear(64,32), nn.SiLU(), nn.Linear(32,4))

    def forward(self, x):
        return self.net(x)


def loss(net, x, gain, state):
    z = net(x)
    return (nn.functional.smooth_l1_loss(z[:,0],gain,beta=.025)
            + .2*nn.functional.cross_entropy(z[:,1:],state))
