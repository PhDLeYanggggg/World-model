"""Hindsight bounds for conservative utility at unchanged stepwise radii."""

import numpy as np


def rotate(cv, anchor, angle):
    x, y = np.moveaxis(anchor-cv, -1, 0)
    c, s = np.cos(angle), np.sin(angle)
    return cv+np.stack((x*c-y*s, x*s+y*c), axis=-1)


def stepwise_oracle(cv, anchor, target, valid, limit):
    d, y = anchor-cv, np.where(valid[..., None], target-cv, 0.)
    cross = d[..., 0]*y[..., 1]-d[..., 1]*y[..., 0]
    angle = np.arctan2(cross, (d*y).sum(-1)).clip(-limit, limit)
    p = rotate(cv, anchor, angle)
    return np.where(valid[..., None], p, anchor)


def choose_rowwise(anchor, paths, target, valid):
    """The paired lower objective adds across agents, so this is exact."""
    q = np.asarray(paths)
    g = np.linalg.norm(anchor-target, axis=-1)-np.linalg.norm(q-target, axis=-1)
    debit = np.linalg.norm(q-anchor, axis=-1)
    lower = np.where(valid, g, -debit).sum(-1)
    return q[lower.argmax(0), np.arange(len(anchor))]
