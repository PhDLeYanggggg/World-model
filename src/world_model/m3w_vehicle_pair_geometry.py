"""Symmetric past-only geometry for a controlled vehicle-pair ablation."""

import numpy as np
from torch import nn

from src.world_model.m3w_vehicle_pair_learning import PairValue


class GeometryValue(PairValue):
    def __init__(self):
        super().__init__(joint=True)
        self.head[0] = nn.Linear(76, 32)


def shuffle_origins(origins, counts, seed):
    rng = np.random.default_rng(seed)
    out = origins.copy()
    for hi, n in zip(counts.cumsum(), counts, strict=True):
        lo = hi-n
        out[lo:hi] = origins[lo+rng.permutation(n)]
    return out


def spacing(m, origins, a, b, ka, kb):
    """No future labels: positions, observed motion, and causal candidate paths."""
    diff = origins[a]-origins[b]
    unit = np.maximum(.5*(m['units'][a]+m['units'][b]), 1e-12)
    v = m['speed'][a]-m['speed'][b]
    vv = (v*v).sum(1)
    dot = (diff*v).sum(1)
    distance = np.linalg.norm(diff, axis=1)
    direction = -dot/np.maximum(distance*np.sqrt(vv), 1e-12)
    closest = np.clip(-dot/np.maximum(vv, 1e-12), 0., 12.)
    near = np.linalg.norm(diff+closest[:, None]*v, axis=1)
    past = np.linalg.norm(diff[:, None]+m['history'][a]-m['history'][b], axis=2).min(axis=1)
    proposed = np.linalg.norm(diff[:, None]+m['paths'][a, ka]-m['paths'][b, kb], axis=2).min(axis=1)
    return np.column_stack((np.log1p(distance/unit), np.log1p(past/unit), direction, closest/12.,
                            np.log1p(near/unit), np.log1p(proposed/unit))).astype(np.float32)
