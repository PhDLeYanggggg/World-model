"""Match two-edit candidate utility against an observed additive anchor."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_vehicle_pair_learning import merit as dual_merit
from src.world_model.m3w_reference_bridge_policy import WIDTH


def alternatives(rows, first, second, anchor):
    """The exact anchor is excluded; keeping an actor may change its family."""
    return ((np.isin(rows[first], rows[anchor]) | np.isin(rows[second], rows[anchor]))
            & ~((first == anchor[0]) & (second == anchor[1])))


class AnchorValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(WIDTH, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.slate = nn.Sequential(nn.Linear(70, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(97, 32), nn.SiLU(), nn.Linear(32, 6))

    def encode(self, x, y, relative):
        u, v = self.actor(x), self.actor(y)
        return self.slate(torch.cat((.5*(u+v), (u-v).abs(), relative), dim=1))

    def forward(self, a, b, rel, anchor_a, anchor_b, anchor_rel, common):
        u, v = self.encode(a, b, rel), self.encode(anchor_a, anchor_b, anchor_rel)
        return self.head(torch.cat((u, v, u-v, common), dim=1))


def loss(net, batch, relative):
    *inputs, target = batch
    p = net(*inputs)
    score = F.binary_cross_entropy_with_logits(p[:, :2], target[:, :2]) + F.smooth_l1_loss(
        p[:, 2:4], target[:, 2:4], beta=.1)
    if relative:
        score = score + F.smooth_l1_loss(p[:, 4:], target[:, 4:], beta=.1)
    return score


def merit(p, relative):
    base = dual_merit(p[:, :4])
    return base + p[:, 5] + np.minimum(p[:, 4], 0.) if relative else base
