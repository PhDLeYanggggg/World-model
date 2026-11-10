"""Pointwise and utility-aware scoring of the same finite shortlist."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_vehicle_action_value as action
from src.world_model import m3w_vehicle_nominee_rank as nominee


K = 3


def order(value):
    score, _ = action.merit(value)
    idx = np.argsort(-score.reshape(-1), kind='stable')[:K]
    i, j = np.unravel_index(idx, score.shape)
    return np.column_stack((i, j+1)), score.reshape(-1)[idx]


def network():
    return nn.Sequential(nn.Linear(nominee.WIDTH, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 6))


def loss(net, batch, utility):
    x, y = batch
    z = net(x)
    a = nn.functional.binary_cross_entropy_with_logits(z[:, 0], y[:, 0])
    if not utility:
        return a
    b = nn.functional.binary_cross_entropy_with_logits(z[:, 1:3], y[:, 1:3])
    c = torch.cat((z[:, 3:5].sigmoid(), z[:, 5:].tanh()), dim=1)
    return a+.5*b+.5*nn.functional.smooth_l1_loss(c, y[:, 3:], beta=.15)


def scores(z, envelope):
    if z.shape[:-1] != envelope.shape or z.shape[-1] != 6 or (envelope < 0.).any():
        raise ValueError('Missing causal path envelope')
    p = 1./(1.+np.exp(-z[..., :5].clip(-80., 80.)))
    utility = envelope*(p[..., 3]-p[..., 4]+.5*np.tanh(z[..., 5])
                        -.5*(1.-p[..., 1])-.25*(1.-p[..., 2]))
    return p[..., 0], utility
