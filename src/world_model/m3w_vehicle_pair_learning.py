"""Direct two-action risk and utility prediction from observed features."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_reference_bridge_policy import WIDTH


class PairValue(nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint
        self.actor = nn.Sequential(nn.Linear(WIDTH, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(70 if joint else 32, 32), nn.SiLU(), nn.Linear(32, 4))

    def forward(self, a, b, relative):
        u, v = self.actor(a), self.actor(b)
        if not self.joint:
            return .5*(self.head(u)+self.head(v))
        return self.head(torch.cat((.5*(u+v), (u-v).abs(), relative), dim=1))


def loss(net, batch):
    a, b, relative, targets = batch
    pred = net(a, b, relative)
    return F.binary_cross_entropy_with_logits(pred[:, :2], targets[:, :2]) + F.smooth_l1_loss(
        pred[:, 2:], targets[:, 2:], beta=.1)


def merit(pred):
    return pred[:, 2:].min(axis=1) - .25*np.logaddexp(0., -pred[:, :2]).sum(axis=1)


def labels(i, j, risk, support, gain, signed_cap, ols_offset, scale):
    """Supervision only: full-record signed cap and OLS offset never become inputs."""
    net = gain[i]+gain[j]
    safe = ((support[i]+support[j] > 0.)
            & ((risk[i]+risk[j]) <= [0., 0., signed_cap]).all(axis=1))
    dual = safe & (net[:, 0] > 0.) & (net[:, 1]+ols_offset > 0.)
    return np.column_stack((safe, dual, net/scale)).astype(np.float32)
