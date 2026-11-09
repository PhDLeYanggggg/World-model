"""Past-only slate features and a lower bound on non-additive pair gain."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_slate_relative_gain as relative

WIDTH = relative.WIDTH + 24


def interaction_lower(cv, paths, recipients, target, valid):
    """Isolate the gain not explained by the two singleton edits.

    For the edited pair, the four-way contrast is
    error(A)+error(B)-error(AB)-error(CV). Grouping the two
    differences by either edit bounds its absolute value by twice
    the smaller displacement. All other pairs cancel.
    """
    i, j = recipients.T
    row = np.arange(len(paths))
    u, v = paths[row, i]-cv[i], paths[row, j]-cv[j]
    mask = valid[i] & valid[j]
    z = cv[i]-cv[j]-np.where(mask[..., None], target[i]-target[j], 0.)
    norm = lambda x: np.linalg.norm(x, axis=-1)
    known = norm(z+u)+norm(z-v)-norm(z+u-v)-norm(z)
    bound = 2.*np.minimum(norm(u), norm(v))
    return np.where(mask, known, -bound).mean(1)


def describe(h, cv, ols, paths, recipients, cutoff):
    i, j = recipients.T
    row = np.arange(len(paths))
    u, v = paths[row, i]-cv[i], paths[row, j]-cv[j]
    origin = h[i, -1]-h[j, -1]
    x, _, _, unit, _ = relative.features(h, cv, ols, paths, cv)
    norm = lambda z: np.linalg.norm(z, axis=-1)/unit
    before = origin[:, None]+cv[i]-cv[j]
    nu, nv = norm(u), norm(v)
    credit_bound = 2.*np.minimum(nu, nv).mean(1)
    gain_bound = norm(paths-cv).mean((1, 2))
    groups = (np.minimum(nu, nv), np.maximum(nu, nv), norm(u-v), norm(before),
              norm(before+u-v), .5*(norm(before+u)+norm(before-v)))
    blocks = [np.log1p(g.reshape(len(g), 3, 4).mean(2)) for g in groups]
    past = norm(h[i]-h[j])
    static = np.column_stack((past[:, -1], past.min(1), past.mean(1), norm(np.diff(h[i]-h[j], axis=1)).mean(1),
                              np.full(len(paths), cutoff/unit), credit_bound))
    z = np.column_stack((x, *blocks, np.log1p(static))).astype(np.float32)
    if z.shape[1] != WIDTH or not np.isfinite(z).all():
        raise ValueError("Invalid causal pair features")
    return z, gain_bound.astype(np.float32), credit_bound.astype(np.float32), unit


class PairValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(WIDTH, 64), nn.SiLU(), nn.Linear(64, 32),
                                  nn.SiLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.head(x)


def values(z, gain_bound, credit_bound):
    """Bound conditional magnitude separately from qualification probability."""
    return z[:, 0].sigmoid()*gain_bound, z[:, 1].tanh()*credit_bound
