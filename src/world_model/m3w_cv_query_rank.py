"""Within-query regret ranking for a frozen causal trajectory proposal."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_nested_velocity_selection import features
from src.world_model.m3w_ols_replacement import velocity


ARMS = ("ols4_row", "ols4_rank", "cv_row", "cv_rank")


def feature_matrix(past, ids, delta, ols_minus_cv):
    """Both references are available to all treatments."""
    x = features(past, ids, delta)
    basis, scale = past[-2][ids], past[-1][ids]
    cv_delta = delta + ols_minus_cv[ids]
    local = np.einsum("nti,nji->ntj", cv_delta[:, [0, 2, 5, 11]], basis) / scale[:, None, None]
    return np.column_stack((x, local.reshape(-1, 8))).clip(-32., 32.).astype(np.float32)


def pairs(queries, legal, sites, seed, per_query=16):
    rng = np.random.default_rng(seed+191)
    out, source = [], []
    for q in queries:
        usable = q[legal[q]]
        if len(usable) < 2:
            continue
        if len(set(sites[usable])) != 1:
            raise ValueError("A query cannot cross source groups")
        left = rng.integers(len(usable), size=per_query)
        right = rng.integers(len(usable)-1, size=per_query)
        right += right >= left
        out.append(np.stack((usable[left], usable[right]), axis=1))
        source.extend([sites[usable[0]]]*per_query)
    if not out:
        raise ValueError("No legal training pairs")
    return np.concatenate(out), np.asarray(source)


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 1))


def pair_loss(prediction, outcome, rank):
    if not rank:
        return torch.nn.functional.smooth_l1_loss(prediction, outcome, beta=.02)
    gap = outcome[:, 0]-outcome[:, 1]
    margin = (prediction[:, 0]-prediction[:, 1])/.05
    return (gap.abs()*torch.nn.functional.softplus(-gap.sign()*margin)).mean()


def prefix_control(history):
    """Past-only preference for OLS4 over CV on two short prefix replays."""
    out = []
    for end in (4, 5):
        past = history[:, :end]
        relative = history[:, end:]-past[:, -1:]
        t = np.arange(1., 9-end)[None, :, None]
        ols = velocity(past, 4)[:, None]*t
        cv = np.diff(past[:, -2:], axis=1)*t
        out.append((np.linalg.norm(cv-relative, axis=-1)-np.linalg.norm(ols-relative, axis=-1)).mean(1))
    return np.mean(out, axis=0)
