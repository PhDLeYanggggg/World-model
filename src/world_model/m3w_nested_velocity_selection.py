"""Causal features and fixed-budget gain/harm selection."""

from itertools import combinations

import numpy as np
import torch
from torch import nn


ARMS = ("gain", "gain_harm")


def folds(sources):
    groups = tuple(sorted(set(sources)))
    return {str(g): {str(h): tuple(sorted((str(g), str(h)))) for h in groups if h != g} for g in groups}


def donor_folds(sources):
    return tuple(combinations(sorted(set(map(str, sources))), 2))


def features(past, row, delta):
    ego, neighbors, mask, basis, scale = (x[row] for x in past)
    h = np.einsum("nti,nji->ntj", delta[:, [0, 2, 5, 11]], basis) / scale[:, None, None]
    summary = (neighbors * mask[..., None]).sum(1) / mask.sum(1).clip(1)[:, None]
    return np.column_stack((ego, summary, mask.sum(1), h.reshape(-1, 8))).clip(-32, 32).astype(np.float32)


def targets(ref, delta, truth, valid, scale):
    """Gain lower bound and row positive-harm upper bound, training only."""
    clean = np.where(valid[..., None], truth, 0.)
    diff = np.linalg.norm(ref-clean, axis=2)-np.linalg.norm(ref+delta-clean, axis=2)
    dist = np.linalg.norm(delta, axis=2)
    lower = np.where(valid, diff, -dist).mean(1)/(12*scale)
    return np.column_stack((lower, np.maximum(-lower, 0.))).astype(np.float32)


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 2))


def loss(net, x, labels, arm):
    value = net(x)
    expected_gain = value[:, 0]
    expected_harm = torch.nn.functional.softplus(value[:, 1])
    out = torch.nn.functional.smooth_l1_loss(expected_gain, labels[:, 0], beta=.02)
    if arm == "gain_harm":
        out += torch.nn.functional.smooth_l1_loss(expected_harm, labels[:, 1], beta=.02)
    return out


def selection_scores(net, x, arm, scale):
    value = net(x)
    risk = torch.nn.functional.softplus(value[:, 1]) if arm == "gain_harm" else 0.
    return 12*scale*(value[:, 0]-risk)


def choose(values, legal, queries, *, optional=False):
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite rankings")
    selected = np.zeros(len(legal), dtype=bool)
    for q in queries:
        if len(q) <= 1:
            continue
        usable = q[legal[q]]
        if len(usable):
            i = usable[values[usable].argmax()]
            selected[i] = not optional or values[i] > 0.
    return selected
