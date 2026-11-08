"""Static road-mask descriptors for the fixed pedestrian path menu."""

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def world_to_pixel(xy, world_file):
    a, d, b, e, c, f = np.asarray(world_file, dtype=float)
    matrix = np.array([[a, b], [d, e]])
    return (xy-np.array([c, f])) @ np.linalg.inv(matrix).T


class RoadDistance:
    def __init__(self, mask, world_file):
        self.world = np.asarray(world_file, dtype=float)
        a, d, b, e, _, _ = self.world
        col, row = np.array([a, d]), np.array([b, e])
        sampling = np.linalg.norm([row, col], axis=1)
        self.axis_cosine = float(abs(row @ col)/sampling.prod())
        if self.axis_cosine > .06 or not mask.any() or mask.all():
            raise ValueError("Need a populated near-orthogonal road raster")
        self.value = (distance_transform_edt(~mask, sampling=sampling).astype(np.float32)
                      - distance_transform_edt(mask, sampling=sampling).astype(np.float32))

    def sample(self, xy):
        uv = world_to_pixel(xy, self.world).reshape(-1, 2)
        out = map_coordinates(self.value, uv.T[::-1], order=1, mode="constant", cval=np.nan)
        return out.reshape(xy.shape[:-1]).astype(float)


def choose(history, bank, on, raster, fallback):
    """Stay at the same road distance, or extrapolate the observed trend."""
    h = raster.sample(history)
    q = raster.sample(history[:, -1:, None]+bank)
    supported = on & np.isfinite(h).all(1) & np.isfinite(q).all((1, 2))
    last = h[:, -1:]
    w = np.arange(4)-1.5
    slope = (h[:, -4:] @ w)/(w @ w)
    speed = np.linalg.norm(np.diff(history[:, -4:], axis=1), axis=-1).mean(1)
    trend = last+np.clip(slope, -speed, speed)[:, None]*np.arange(1, 13)[None]
    losses = {"map_persistence": np.abs(q-last[:, None]).mean(2),
              "map_continuation": np.abs(q-trend[:, None]).mean(2)}
    res = {}
    idx = np.arange(len(h))
    for k, loss in losses.items():
        a = np.where(supported[:, None], loss, np.inf).argmin(1)
        res[k] = np.where(supported & (loss[idx, a]+1e-10 < loss[:, fallback]), a, fallback)
    return res, supported, h
