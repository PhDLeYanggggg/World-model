"""Support targets for a fixed two-edit policy under the original risk rule."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def prefix(rows, query, first, first_features, first_gain, easy, scale):
    at = query[rows[0]]
    known = first is not None and query[first] <= at
    if not known:
        return np.zeros(first_features.shape[1]+5, dtype=np.float32)
    age = at-query[first]
    return np.r_[first_features[first], 1., float(age == 0), np.log1p(age),
                 np.clip(first_gain[first]/scale[first], -8., 8.), easy[first]].astype(np.float32)


def support_targets(gain, excess, easy, base, ref_easy, first=None, ols_offset=0.):
    """Combine the small edit with one distinct, already fixed large edit.

    Each component is (two gains, three excesses, easy reference, all reference).
    The first edit can be later than the query in labels, but not in features.
    """
    offset_gain, offset_risk, selected_easy, selected_base = first if first is not None else (
        np.zeros(2), np.zeros(3), 0., 0.)
    total = gain+offset_gain+np.array([0., ols_offset])
    risk = ((base+selected_base > 0.) & (easy+selected_easy > 0.) & (ref_easy > 0.)
            & ((excess+offset_risk) <= [0., 0., .02*ref_easy]).all(1))
    return risk, risk & (total > 0.).all(1)


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 3))


def loss(net, x, y):
    z = net(x)
    return F.smooth_l1_loss(z[:, 0], y[:, 0], beta=.05) + F.binary_cross_entropy_with_logits(z[:, 1:], y[:, 1:])


def choose(rows, anchor, z, guard=False):
    if z.shape != (len(rows), 3) or anchor not in rows or not np.isfinite(z).all():
        raise ValueError("Invalid fixed support menu")
    i = int(np.flatnonzero(rows == anchor)[0])
    if guard:
        ok = (z[:, 0] >= z[i, 0]) & (z[:, 1] >= z[i, 1]) & (z[:, 2] > z[i, 2])
        return int(rows[np.where(ok, z[:, 2], -np.inf).argmax()]) if ok.any() else anchor
    return int(rows[z[:, 2].argmax()])
