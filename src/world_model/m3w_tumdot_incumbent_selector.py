"""Direct pairwise supervision for fixed-budget shape replacement."""

import numpy as np
from torch.nn import functional as F


def labels(cv, bank, y, valid, radius, easy_cut, incumbent, paired):
    floor = bank[:, incumbent]
    shape = radius[:, None]
    cv_gain = np.column_stack([paired(cv, p, y, valid)[1].mean(1) for p in bank.swapaxes(0, 1)])/shape
    pair_gain = np.column_stack([paired(floor, p, y, valid)[1].mean(1) for p in bank.swapaxes(0, 1)])/shape
    safe_y = np.where(valid[..., None], y, 0.)
    ade = lambda p: (np.linalg.norm(p-safe_y, axis=-1)*valid).sum(1)/valid.sum(1).clip(1)
    ref = ade(cv)
    harm = np.column_stack([np.maximum(ade(p)-ref, 0.) for p in bank.swapaxes(0, 1)])
    diff = (harm-harm[:, incumbent, None])/shape
    easy = (ref > 0.) & (ref <= easy_cut)
    delta = np.stack((diff, diff*easy[:, None]), axis=-1)
    return {k: np.concatenate((v[..., None], delta), axis=-1).astype(np.float32)
            for k, v in (("cv_target", cv_gain), ("pair_target", pair_gain))}


def centered(net, x, incumbent):
    s = net(x)
    return s-s[:, incumbent, None]


def loss(net, x, y, incumbent, pair_target):
    raw = net(x)
    gain = raw[..., 0]-raw[:, incumbent, None, 0] if pair_target else raw[..., 0]
    harm = raw[..., 1:]-raw[:, incumbent, None, 1:]
    return F.mse_loss(gain, y[..., 0])+.5*F.mse_loss(harm, y[..., 1:])


def choices(score, incumbent, guard=False):
    value = score[..., 0].copy()
    if guard:
        value[(score[..., 1:] > 0.).any(2)] = -np.inf
    value[:, incumbent] = 0.
    return value.argmax(1)
