"""Forecast-independent edited-easy support from causal CV consistency."""

import numpy as np
from torch import nn

from src.world_model import m3w_av2_history_replay as history


def inputs(d, cutoff):
    h, cv, ols = (d[k] for k in ("h", "cv", "ols4"))
    speed = np.linalg.norm(np.diff(h, axis=1), axis=-1)
    scale = speed.mean(1).clip(.05)
    error = np.linalg.norm(ols-cv, axis=-1)
    prefix = []
    for k in history.PREFIXES:
        n = 8-k
        a, b, _, _ = history.motion.rollout(h[:, :k], n)
        y = h[:, k:]-h[:, k-1:k]
        e = np.linalg.norm(a-y, axis=-1)
        prefix.extend((e.mean(1)/scale, e.max(1)/scale,
                       np.linalg.norm(b-y, axis=-1).mean(1)/scale))
    return np.column_stack((speed/scale[:, None], np.log1p(scale), np.log1p(cutoff/scale),
        error.mean(1)/scale, error.max(1)/scale, *prefix)).clip(-24., 24.).astype(np.float32)


def labels(cv, y, valid, cutoff):
    safe = np.where(valid[..., None], y, 0.)
    e = np.where(valid, np.linalg.norm(cv-safe, axis=-1), 0.).sum(1)/valid.sum(1).clip(1)
    return (e > 0.) & (e <= cutoff)


def head(width):
    return nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))


def either(prob):
    return 1.-np.prod(1.-prob, axis=-1)
