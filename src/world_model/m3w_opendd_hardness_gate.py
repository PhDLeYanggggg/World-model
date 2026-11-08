"""Predict CV difficulty independently of the candidate neural trajectory."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_donor_motion_shape import motion_frame
from src.world_model.m3w_ols_replacement import rollout


def features(history):
    v, _, scale = motion_frame(history)
    err = []
    for k in (4, 5, 6):
        cv, ols, _, _ = rollout(history[:, :k], 8-k)
        observed = history[:, k:]-history[:, k-1:k]
        err.extend(np.log1p(np.linalg.norm(p-observed, axis=-1).mean(1)/scale) for p in (cv, ols))
    return np.c_[v, np.log1p(scale), *err].clip(-16., 16.).astype(np.float32)


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 3))


def loss(net, x, labels):
    return nn.functional.cross_entropy(net(x), labels)


def decide(p, legal):
    return legal & (p[:, 2] >= .50) & (p[:, 0] < .25)


def matched_count(query, selected, eligible, score):
    """Causal within-query rank control with exactly the learned switch count."""
    if (selected & ~eligible).any():
        raise ValueError("Selected row is not eligible")
    out = np.zeros(len(query), dtype=bool)
    for q in np.unique(query[selected]):
        in_query = query == q
        count = (in_query & selected).sum()
        rows = np.flatnonzero(in_query & eligible)
        out[rows[np.argsort(-score[rows], kind="stable")[:count]]] = True
    np.testing.assert_array_equal(np.bincount(query[out], minlength=query.max()+1),
                                  np.bincount(query[selected], minlength=query.max()+1))
    return out
