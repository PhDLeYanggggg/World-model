"""Match a target displacement sum without reordering the residual path."""

import numpy as np


def amplitude_control(cv, prediction, target):
    cv, prediction, target = (np.asarray(x, dtype=float) for x in (cv, prediction, target))
    if (cv.shape != prediction.shape or cv.shape != target.shape or cv.ndim != 3 or cv.shape[-1] != 2
            or not all(np.isfinite(x).all() for x in (cv, prediction, target))):
        raise ValueError('Expected three finite aligned forecast arrays')
    r = prediction-cv
    a = np.linalg.norm(r, axis=-1)
    radius = a.max(1, keepdims=True)
    mutable = (a > 0.) & (a < radius)
    mutable[:, -1] = False
    diff = np.linalg.norm(target-cv, axis=-1).sum(1)-a.sum(1)
    down = diff <= 0.
    capacity = np.where(down[:, None], a, radius-a)*mutable
    total = capacity.sum(1)
    possible = np.abs(diff) <= total+1e-12
    weight = np.clip(np.divide(np.abs(diff), total, out=np.zeros_like(diff), where=total > 0.), 0., 1.)
    new = a+np.sign(diff)[:, None]*weight[:, None]*capacity
    ratio = np.divide(new, a, out=np.ones_like(a), where=a > 0.)
    control = np.where((mutable & possible[:, None])[..., None], cv+r*ratio[..., None], prediction)
    return control, possible
