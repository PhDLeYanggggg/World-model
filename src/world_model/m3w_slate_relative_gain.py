"""Predict matched-slate relative motion gain without reading future masks."""

import numpy as np
import torch
from torch import nn

WIDTH = 33


def features(history, cv, ols, candidate, reference):
    """Encode all relative paths, including pairs with an unchanged recipient."""
    i, j = np.triu_indices(len(history), 1)
    origin = history[i, -1] - history[j, -1]
    motion = np.diff(history, axis=1)
    unit = max(float(np.linalg.norm(motion, axis=-1).mean()), .05)
    a = (candidate[:, i] - candidate[:, j]) / unit
    b = (reference[i] - reference[j])[None] / unit
    v = (cv[i] - cv[j]) / unit
    o = (ols[i] - ols[j]) / unit
    gap = np.linalg.norm(a - b, axis=-1)
    bound = gap.mean((1, 2))

    def blocks(x):
        return x.reshape(len(x), len(i), 3, 4).mean((1, 3))

    def static(x):
        return np.broadcast_to(x, (len(a), len(x)))

    # Log distances are invariant to translation and rotation and keep distant pairs bounded.
    origin_dist = np.linalg.norm(origin, axis=-1) / unit
    speed = np.linalg.norm(motion[i] - motion[j], axis=-1) / unit
    ctx = static(np.log1p([len(history), origin_dist.min(), origin_dist.mean(),
                          speed.mean(), speed[:, -1].mean(), np.linalg.norm(v[:, -1], axis=-1).mean()]))
    av, bv = np.linalg.norm(a - v, axis=-1), np.linalg.norm(b - v, axis=-1)
    ao, bo = np.linalg.norm(a - o, axis=-1), np.linalg.norm(b - o, axis=-1)
    final_a = np.linalg.norm(a + origin[None, :, None] / unit, axis=-1)
    final_b = np.linalg.norm(b + origin[None, :, None] / unit, axis=-1)
    symmetric = np.column_stack((ctx, np.log1p(blocks(gap)), np.log1p(blocks(av+bv)),
                                 np.log1p(blocks(ao+bo)), np.log1p(blocks(final_a+final_b))))
    signed = np.column_stack((blocks(bv-av), blocks(bo-ao), blocks(final_b-final_a),
                              np.log1p(blocks(av))-np.log1p(blocks(bv)),
                              np.log1p(blocks(ao))-np.log1p(blocks(bo))))
    x = np.column_stack((symmetric, signed)).astype(np.float32)
    xr = np.column_stack((symmetric, -signed)).astype(np.float32)
    return x, xr, bound.astype(np.float32), unit, (bo-ao).mean((1, 2))*unit


def labels(candidate, reference, target, valid, unit):
    """Known gain contribution and a nonnegative missing-pair debit, both per grid step."""
    i, j = np.triu_indices(len(reference), 1)
    mask = valid[i] & valid[j]
    actual = np.where(mask[..., None], target[i]-target[j], 0.)
    a, b = candidate[:, i]-candidate[:, j], reference[i]-reference[j]
    diff = np.linalg.norm(b-actual, axis=-1) - np.linalg.norm(a-actual, axis=-1)
    known = np.where(mask, diff, 0.).mean((1, 2))
    debit = np.where(mask, 0., np.linalg.norm(a-b, axis=-1)).mean((1, 2))
    return (np.column_stack((known, debit)) / unit).astype(np.float32)


class RelativeGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(WIDTH, 48), nn.SiLU(), nn.Linear(48, 24),
                                   nn.SiLU(), nn.Linear(24, 2))

    def forward(self, x, xr, bound):
        a, b = self.score(x), self.score(xr)
        known = torch.tanh((a[:, 0]-b[:, 0])*.5) * bound
        debit = torch.sigmoid((a[:, 1]+b[:, 1])*.5) * bound
        return torch.stack((known, debit), dim=-1)
