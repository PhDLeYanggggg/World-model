"""Risk estimation and count-matched selection on fixed causal proposals."""
import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.m3w_cv_anchor_budget import risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors


def proposals(gain, risk, legal, anchor):
    score = np.where(legal, gain.min(-1) - 2. * risk.clip(0.).sum(-1), -np.inf)
    out = np.where(legal.any(1), score.argmax(1), -1)
    selected = anchor >= 0
    if not legal[np.flatnonzero(selected), anchor[selected]].all():
        raise ValueError('Parent selected an illegal proposal')
    out[selected] = anchor[selected]
    return out


def targets(cv, path, y, valid, scale, cut):
    c, _ = errors(cv, y, valid)
    p, _ = errors(path, y, valid)
    easy = valid.any(1) & (c > 0.) & (c <= cut)
    mass = np.column_stack((np.where(valid.any(1), c, 0.), np.where(easy, c, 0.)))
    excess = risk_excess(cv, path[:, None], y, valid, cut)[:, 0]
    signed = np.where(easy, p - c, 0.)
    return (np.column_stack(((excess + .02 * mass).clip(0.), signed)) / scale[:, None]).astype(np.float32)


def decode(z):
    return torch.cat((F.softplus(z[:, :2] - 4.), z[:, 2:]), 1)


def loss(net, x, y, quantile):
    error = y - decode(net(x))
    return torch.maximum(quantile * error, (quantile - 1.) * error).mean()


def rank(gain, predicted, ref, scale, penalty):
    risk = predicted[:, :2] - .02 * ref / scale[:, None]
    signed = predicted[:, 2] - .02 * ref[:, 1] / scale
    return gain.min(1) / scale - penalty * (risk.clip(0.).sum(1) + signed.clip(0.))


def select(choice, anchor, score, groups, *, margin=None):
    out = np.full_like(choice, -1)
    for q in groups:
        k = (anchor[q] >= 0).sum()
        legal = q[choice[q] >= 0]
        order = legal[np.argsort(-score[legal], kind='stable')[:k]]
        if margin is not None:
            order = order[score[order] > margin]
        elif len(order) != k:
            raise ValueError('Could not match original intervention count')
        out[order] = choice[order]
    return out


def upper_at_count(gain, choice, anchor, groups, weights):
    """Upper utility for each comparator separately, with no risk constraints."""
    totals = np.zeros(gain.shape[1])
    for q in groups:
        k = (anchor[q] >= 0).sum()
        if k:
            q = q[choice[q] >= 0]
            if len(q) < k:
                raise ValueError('Oracle cannot match count')
            v = gain[q] * weights[q, None]
            totals += np.sort(v, axis=0)[-k:].sum(0)
    return totals
