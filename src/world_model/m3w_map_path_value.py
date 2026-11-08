"""Map features and simultaneous-query control for a matched path-value head."""

import numpy as np
from torch import nn

from src.world_model import m3w_matched_path_choice as cost

MAP_WIDTH = 16
MODES = ("no_map", "map", "wrong_map")


def map_features(history, cv, bank, radius, raster):
    origin = history[:, -1:]
    h = raster.sample(history)
    c = raster.sample(origin+cv)
    p = raster.sample(origin[:, None]+bank)
    support = np.isfinite(h).all(1) & np.isfinite(c).all(1) & np.isfinite(p).all((1, 2))
    speed = np.sqrt(np.square(np.diff(history, axis=1)).sum(2).mean(1))
    if (speed <= 0).any() or (radius <= 0).any():
        raise ValueError("Require active moving histories and common radii")
    h, c, p = (np.where(support.reshape((-1,)+(1,)*(z.ndim-1)), z, 0.) for z in (h, c, p))
    scale = speed[:, None]
    w = np.arange(4)-1.5
    v = np.clip((h[:, -4:] @ w)/(w @ w), -speed, speed)
    trend = h[:, -1:]+v[:, None]*np.arange(1, 13)[None]
    chunk = lambda a: a.reshape((*a.shape[:-1], 3, 4)).mean(-1)
    common = np.column_stack((support, np.arcsinh(h[:, -1]/speed), (h[:, -4:]-h[:, -1:])/scale,
                              (h.min(1) < 0) & (h.max(1) > 0)))
    delta = chunk((p-c[:, None])/radius[..., None])
    resid = chunk(np.abs(p-trend[:, None])/scale[:, None])
    signed = chunk(np.arcsinh(p/scale[:, None]))
    x = np.concatenate((np.broadcast_to(common[:, None], (*radius.shape, common.shape[1])), delta, resid, signed), axis=2)
    if x.shape[-1] != MAP_WIDTH or not np.isfinite(x).all():
        raise ValueError("Unexpected map features")
    return x.astype(np.float32), support


def wrong_map(features, queries, supported):
    """Exchange map tensors only between mapped agents in the same query."""
    wrong = np.zeros_like(features)
    correctly_mapped = np.zeros_like(features)
    exchanged = np.zeros(len(features), dtype=bool)
    for q in np.unique(queries[supported]):
        ids = np.flatnonzero((queries == q) & supported)
        if len(ids) >= 2:
            correctly_mapped[ids] = features[ids]
            wrong[ids] = features[np.roll(ids, 1)]
            exchanged[ids] = True
    return correctly_mapped, wrong, exchanged


class PathValue(cost.ActionValue):
    def __init__(self):
        super().__init__()
        self.value[0] = nn.Linear(41+MAP_WIDTH, 48)
