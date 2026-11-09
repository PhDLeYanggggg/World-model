"""Matched clean and synthetically perturbed donor inputs."""
import numpy as np


def perturb(history, future, rng):
    """Change the observation origin, never the absolute forecast target."""
    h = np.asarray(history, dtype=float)
    y = np.asarray(future, dtype=float)
    speed = np.sqrt(np.square(np.diff(h, axis=1)).sum(2).mean(1))
    strength = rng.choice([0., .25, .75], size=len(h))
    z = rng.standard_normal(h.shape)
    for t in range(1, h.shape[1]):
        z[:, t] = .5*z[:, t-1] + np.sqrt(.75)*z[:, t]
    shift = (speed*strength)[:, None, None]*z/np.sqrt(2.)
    return h+shift, y-shift[:, -1:]


def normalized(history, future=None):
    """Only past observations determine the shared length normalization."""
    h = np.asarray(history, dtype=float)
    scale = np.sqrt(np.square(np.diff(h, axis=1)).sum(2).mean(1))
    scale = np.where(scale > 1e-8, scale, 1.)
    h = (h-h[:, -1:])/scale[:, None, None]
    return h, scale, None if future is None else future/scale[:, None, None]
