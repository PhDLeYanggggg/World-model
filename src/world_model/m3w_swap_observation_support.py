"""Separate future support and signed observed effects for one frozen swap."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def footprint(history, offsets, cv, drop, add):
    """Past-only occupied rectangle: a support proxy, not a camera calibration."""
    past = history+offsets[:, None]
    lo, hi = past.min((0, 1)), past.max((0, 1))
    span = (hi-lo).clip(.1)
    pos = (cv+offsets[:, None]-lo)/span
    margin = np.minimum(pos, 1.-pos).min(-1)
    x = np.column_stack(((margin >= 0.).mean(1), margin[:, -1], margin.min(1)))
    x[:, 1:] = np.clip(x[:, 1:], -8., 1.)
    return np.column_stack((x[drop], x[add], np.broadcast_to(x.mean(0), (len(drop), 3)))).astype(np.float32)


def targets(y, valid, drop, add, geometry):
    r, p, di, rr, rp, dr, i, j, edge, norm = geometry
    ids = np.column_stack((drop, add))

    def factor(ref, pred, truth, seen, distance, total):
        diff = np.linalg.norm(ref-truth, axis=-1)-np.linalg.norm(pred-truth, axis=-1)
        mass = np.where(seen, distance, 0.).sum((1, 2))
        support = mass/total.clip(1e-12)
        pos = np.where(seen, np.maximum(diff, 0.), 0.).sum((1, 2))/mass.clip(1e-12)
        neg = np.where(seen, np.maximum(-diff, 0.), 0.).sum((1, 2))/mass.clip(1e-12)
        return np.column_stack((support, pos, neg))

    return np.clip(np.stack((factor(r, p, y[ids], valid[ids], di, norm[:, 0]),
                             factor(rr, rp, y[i]-y[j], valid[i] & valid[j] & edge[..., None], dr, norm[:, 1])),
                            axis=1), 0., 1.).astype(np.float32)


def lower(components):
    s, a, b = (components[..., i] for i in range(3))
    return s*(a-b)-(1.-s)


def network(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 48), nn.SiLU(),
                         nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 8), nn.Unflatten(-1, (2, 4)))


def loss(logits, targets):
    pred = logits[..., :3].sigmoid()
    support = F.smooth_l1_loss(pred[..., 0], targets[..., 0], beta=.2)
    cond = F.smooth_l1_loss(pred[..., 1:], targets[..., 1:], beta=.2, reduction='none').mean(-1)
    cond = (cond*targets[..., 0]).sum()/targets[..., 0].sum().clamp_min(1.)
    sign = F.binary_cross_entropy_with_logits(logits[..., 3], (lower(targets) > 0.).to(torch.float32))
    return support + cond + sign
