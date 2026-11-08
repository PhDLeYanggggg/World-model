"""Rank legal recipients without changing the frozen support release clock."""

import numpy as np
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_opendd_two_reserves import EASY_THRESHOLD


def menu(rows, easy, first=None):
    rows = rows[rows != first] if first is not None else rows
    if not len(rows) or easy[rows].max() < EASY_THRESHOLD:
        return rows[:0], None
    return rows, int(rows[easy[rows].argmax()])


def features(x, easy, rows, anchor):
    return np.c_[x[rows], x[rows]-x[anchor], easy[rows], easy[rows]-easy[anchor],
                 np.full(len(rows), np.log1p(len(rows)))].astype(np.float32)


def choose(rows, anchor, easy, scores):
    if anchor not in rows or scores.shape != (len(rows), 3) or not np.isfinite(scores).all():
        raise ValueError("Invalid support menu or scores")
    gain, support, harm = scores.T
    i = int(np.flatnonzero(rows == anchor)[0])
    eligible = easy[rows] >= EASY_THRESHOLD
    safe = (gain >= gain[i]) & (harm <= harm[i])
    best = lambda value, mask: int(rows[np.where(mask, value, -np.inf).argmax()])
    return dict(local_gain=best(gain, np.ones(len(rows), dtype=bool)),
                local_support=best(support, np.ones(len(rows), dtype=bool)),
                restricted_support=best(support, eligible),
                guarded_support=best(support, safe & (support > support[i]))
                if (safe & (support > support[i])).any() else anchor)


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 3))


def loss(net, x, target):
    z = net(x)
    return (F.smooth_l1_loss(z[:, 0], target[:, 0], beta=.05)
            + F.binary_cross_entropy_with_logits(z[:, 1], target[:, 1])
            + F.smooth_l1_loss(F.softplus(z[:, 2]), target[:, 2], beta=.05))
