"""Causal prefix accounting for marginal recording risk."""
import numpy as np
import torch
from torch.nn import functional as F


WIDTH = 12


def marginal(prefix, current):
    """Difference in positive excess versus doing nothing in this query."""
    risk = current[:, 2:5]
    return (prefix[2:5] + risk).clip(0.) - (prefix[2:5] + risk[:1]).clip(0.)


def context(prefix, current, elapsed, n, scale):
    a, e = np.maximum(prefix[6:] + current[0, 6:], 1.)
    s = max(1., elapsed * scale)
    z = np.r_[prefix[:2] / s, prefix[2:5] / [a, e, e], prefix[5] / e,
              np.log1p(prefix[6:]), np.log1p(elapsed)]
    d = marginal(prefix, current) / (n * scale)
    out = np.column_stack((np.broadcast_to(z, (len(current), len(z))), d))
    if out.shape[1] != WIDTH:
        raise ValueError('Unexpected causal state width')
    return np.arcsinh(out).astype(np.float32)


def features(x, prefix, current, elapsed, n, scale, state):
    z = context(prefix, current, elapsed, n, scale)
    return np.concatenate((x, z if state else np.zeros_like(z)), axis=-1)


def target(x, prefix, current, n, scale):
    y = x.copy()
    y[:, 2:5] = marginal(prefix, current) / (n * scale)
    return y


def score(y, price):
    d = y - y[:1]
    risk = d[:, 2:5]
    return d[:, :2].min(1) - price * (risk.clip(0.).sum(1) - .25 * (-risk).clip(0.).sum(1)) + .25 * d[:, 5]


def pick(y, counts, price, margin, count=None):
    v = score(y, price)
    v[1:] -= margin
    if count is not None:
        v = np.where(counts == count, v, -np.inf)
    k = int(v.argmax())
    if not np.isfinite(v[k]):
        raise ValueError('No option with the required exact query count')
    return k


def objective(net, x, y, state):
    if not state:
        x = torch.cat((x[..., :-WIDTH], torch.zeros_like(x[..., -WIDTH:])), dim=-1)
    z = net(x)
    return F.smooth_l1_loss(z, y) + .5 * F.smooth_l1_loss(z[:, 1] - z[:, 0], y[:, 1] - y[:, 0])
