"""Matched OpenDD vehicle K=1 control with the pinned EqMotion core."""

import numpy as np

from src.world_model import m3w_opendd_comoving_eqmotion as eq
from src.world_model.m3w_vehicle_residual import PEAK_BOUND, loss

ARMS = ('comoving_peers', 'comoving_ego')


def inputs(history, origins, counts):
    if (history.shape != (len(origins), 8, 2) or origins.shape != (len(history), 2)
            or len(history) != sum(counts) or np.any(counts <= 0)):
        raise ValueError('Require aligned past-only origins and vehicle histories')
    x = history.astype(np.float64)+origins[:, None]
    ego, neighbors, mask, _ = eq.context(x, np.r_[0, counts.cumsum()])
    return ego, neighbors, mask


def bounded(path, cv, scale):
    delta = path-cv
    limit = PEAK_BOUND*scale
    peak = delta.norm(dim=-1).amax(dim=1)
    return cv+delta*(limit/peak.maximum(limit))[:, None, None]


def objective(net, batch):
    ego, neighbors, mask, cv, scale, y, valid = batch
    predicted = bounded(net(ego, neighbors, mask), cv, scale)
    s = scale[:, None, None]
    return loss(predicted/s, cv/s, y/s, valid, completion=True)
