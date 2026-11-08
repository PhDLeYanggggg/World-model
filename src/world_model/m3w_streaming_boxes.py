"""Current-frame association and past-only image-plane motion forecasts."""

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


def overlap(a, b):
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    area = np.maximum(0., right-left).prod(axis=-1)
    aa = (a[:, 2:]-a[:, :2]).prod(axis=1)
    ab = (b[:, 2:]-b[:, :2]).prod(axis=1)
    return area / (aa[:, None]+ab[None, :]-area).clip(1e-9)


def match(old, new):
    a, b = old[:, :4], new[:, :4]
    iou = overlap(a, b)
    ac, bc = (a[:, :2]+a[:, 2:])/2., (b[:, :2]+b[:, 2:])/2.
    distance = np.linalg.norm(ac[:, None]-bc[None, :], axis=-1)
    diagonal = np.maximum(np.linalg.norm(a[:, 2:]-a[:, :2], axis=1)[:, None],
                          np.linalg.norm(b[:, 2:]-b[:, :2], axis=1)[None, :])
    relative = distance / diagonal.clip(1.)
    valid = ((iou >= .1) | (relative <= .75)) & (old[:, 5, None] == new[:, 5][None])
    cost = .65*(1.-iou)+.35*relative.clip(0., 1.)
    i, j = linear_sum_assignment(np.where(valid, cost, 1e6))
    return list(zip(i[valid[i,j]], j[valid[i,j]], strict=True))


class StreamingTracks:
    def __init__(self):
        self.index = -1
        self.next_id = 0
        self.active = {}

    def step(self, index, detection):
        if index != self.index+1:
            raise ValueError("Track updates must consume every sampled frame in order")
        if (detection.ndim != 2 or detection.shape[1] != 6 or not np.isfinite(detection).all()
                or (detection[:, 2:4] <= detection[:, :2]).any()):
            raise ValueError("Expected finite xyxy, confidence, class")
        self.index = index
        self.active = {k:v for k,v in self.active.items() if index-v[-1][0] <= 2}
        ids = list(self.active)
        lookup = {}
        if ids and len(detection):
            lookup = {j:ids[i] for i,j in match(np.stack([self.active[k][-1][1] for k in ids]), detection)}
        rows, forecasts = [], []
        for j, box in enumerate(detection):
            key = lookup.get(j)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.active[key] = deque(maxlen=8)
            history = self.active[key]
            history.append((index, box.copy()))
            rows.append((key, *box))
            if len(history) == 8 and history[0][0] == index-7:
                hist = np.stack([b[:2]+b[2:4] for _,b in history]) / 2.
                last = hist[-1]
                cv = last-hist[-2]
                ols = np.array([-1.5,-.5,.5,1.5]) @ (hist[-4:]-last) / 5.
                t = np.arange(1., 13.)[:, None]
                forecasts.append((key, last + cv*t, last + ols*t))
        return np.asarray(rows, dtype=float).reshape(-1, 7), forecasts
