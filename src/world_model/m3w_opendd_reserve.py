"""Release at most one additional low-amplitude edit from causal packets."""

import numpy as np
import torch
from torch import nn

AMPLITUDE = .25
JOINT_THRESHOLD = .25
EASY_THRESHOLD = .50


def pick(rows, easy, gain, joint, arm, force=False):
    if not len(rows):
        return None
    if arm == "easy":
        value, use = easy, easy >= EASY_THRESHOLD
    elif arm == "gain":
        value, use = gain, gain > 0.
    elif arm == "joint":
        value, use = joint, (joint >= JOINT_THRESHOLD) & (gain > 0.)
    else:
        raise ValueError("Unknown reserve arm")
    if not np.isfinite(value).all():
        raise ValueError("Non-finite reserve scores")
    if force:
        return int(rows[value.argmax()])
    return int(rows[np.where(use, value, -np.inf).argmax()]) if use.any() else None


def first(packets, arm):
    for t, p in enumerate(packets):
        row = pick(*p, arm)
        if row is not None:
            return t, row
    return None


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 3))


def loss(net, x, target):
    p = net(x)
    return (nn.functional.smooth_l1_loss(p[:, 0], target[:, 0], beta=.05)
            + nn.functional.binary_cross_entropy_with_logits(p[:, 1:], target[:, 1:]))
