"""Past-only peer controls for choosing a single wait per recording."""

import numpy as np
from torch import nn

from src.world_model.m3w_opendd_velocity_candidate import frames

MODES = ("self", "peer", "wrong_peer")
PEER_WIDTH = 34


def peer_features(history, query, anchor):
    """Nearest two peers in the focal observed frame, or another same-query focal."""
    if len(history) != len(query) or history.shape[1:] != (8, 2):
        raise ValueError("Need eight observed positions per query member")
    rot, scale = frames(history)
    offsets = np.r_[0, np.bincount(query).cumsum()]

    def describe(i, members):
        peers = members[members != i]
        rel = history[peers]-history[i]
        xy = (rel@rot[i])/(12.*scale[i])
        dist = np.linalg.norm(rel[:, -1], axis=-1)
        keys = np.vstack((xy.reshape(len(xy), 16).T[::-1], dist))
        indices = np.lexsort(keys)[:2]
        z = np.zeros((2, 17), dtype=np.float32)
        z[:len(indices), :16] = xy[indices].reshape(len(indices), 16).clip(-12., 12.)
        z[:len(indices), 16] = 1.
        return z.ravel()

    aligned, wrong = [], []
    for i in anchor:
        q = query[i]
        ids = np.arange(offsets[q], offsets[q+1])
        j = ids[(i-offsets[q]+1) % len(ids)]
        aligned.append(describe(i, ids))
        wrong.append(describe(j, ids))
    return np.asarray(aligned), np.asarray(wrong)


def opportunity_features(query, legal, radii, cap, release):
    """Summarize opportunities strictly before and at release, never after it."""
    n = len(cap)
    count = np.bincount(query, minlength=n)
    l = np.bincount(query, weights=legal, minlength=n)
    r = np.zeros(n)
    np.maximum.at(r, query, radii)
    cumulative = np.r_[0., l.cumsum()]
    left = np.maximum(0, release-4)
    return np.c_[np.log1p(count[release]), np.log1p(l[release]), np.log1p(r[release]),
                 np.log1p(cumulative[release]-cumulative[left]), cap[release]/2.].astype(np.float32)


class Value(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.net(x)


def loss(net, x, y):
    z = net(x)
    event = nn.functional.binary_cross_entropy_with_logits(z[:, :3], y[:, :3])
    return event + nn.functional.smooth_l1_loss(z[:, 3], y[:, 3], beta=.025)


def causal_pick(release, success, movable, value):
    """Accept the first qualifying query, ranking only simultaneous offers."""
    ix = np.flatnonzero((success >= .25) & (movable >= .5) & (value > 0.))
    if not len(ix):
        return None
    first = ix[release[ix] == release[ix[0]]]
    return int(first[success[first].argmax()])
