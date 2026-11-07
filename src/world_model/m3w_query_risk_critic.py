"""Predict the costs of a proposed switch set from the current query only."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import errors


BUDGET = .02
RIDGE = .04
HEADS = ('gain_lower', 'all_excess', 'easy_excess', 'easy_support')


def proposal_features(history, floor, bank, choice, costs, scale, easy_ref):
    ids = np.flatnonzero(choice >= 0)
    at = ids, choice[ids]
    norm = float(scale[ids].mean())
    q = np.column_stack([c[at] for c in (*costs, easy_ref)]) / norm
    d = bank[at] - floor[ids]
    radius = np.linalg.norm(d, axis=-1).max(1) / norm
    past = np.diff(history, axis=1)
    acceleration = np.linalg.norm(np.diff(past, axis=1), axis=-1).mean(1)
    speed = np.linalg.norm(past, axis=-1).mean(1)
    dist = np.linalg.norm(history[ids, None, -1] - history[None, :, -1], axis=-1)
    dist[np.arange(len(ids)), ids] = np.inf
    nearest = dist.min(1)
    def two(v):
        return np.sort(v) if len(v) == 2 else np.repeat(v, 2)

    compact = np.r_[1., np.arcsinh(q.sum(0))]
    added = np.r_[np.arcsinh(q.min(0)), np.arcsinh(q.max(0)), np.log1p(two(radius)),
                  np.log1p(len(history)), np.log(norm), np.log1p(two(nearest)/norm),
                  np.log1p(two(acceleration[ids])/norm), np.log1p(two(speed[ids])/norm),
                  np.log1p(np.linalg.norm(d[0]-d[-1], axis=-1).max()/norm)]
    return compact, np.r_[compact, added], norm


def action_targets(floor, pred, future, valid, easy_cut):
    known = valid.any(1)
    ref, _ = errors(floor, future, valid)
    err, _ = errors(pred, future, valid)
    ref = np.where(known, ref, 0.)
    gain = np.where(known, ref-err, 0.)
    easy = known & (ref > 0.) & (ref <= easy_cut)
    missing = np.where(known, 0., np.linalg.norm(pred-floor, axis=-1).max(1))
    harm = np.maximum(-gain, 0.)
    return np.column_stack((gain-missing, harm+missing-BUDGET*ref,
                            easy*(harm-BUDGET*ref)+missing, easy))


def targets(actions, norm):
    y = actions.sum(0)
    y[:3] /= norm
    y[3] = float(y[3] > 0.)
    return y


def fit(x, y, groups):
    _, ix, n = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / (len(n)*n[ix])
    mean = w @ x[:, 1:]
    sd = np.sqrt(w @ (x[:, 1:]-mean)**2).clip(.02)
    z = np.column_stack((x[:, 0], (x[:, 1:]-mean)/sd))
    reg = np.eye(z.shape[1])*RIDGE
    reg[0, 0] = 0.
    return np.r_[mean, sd, np.linalg.solve(z.T @ (w[:, None]*z)+reg,
                                          z.T @ (w[:, None]*y)).ravel()]


def predict(x, packet):
    p = x.shape[1]-1
    z = np.column_stack((x[:, 0], (x[:, 1:]-packet[:p])/packet[p:2*p]))
    y = z @ packet[2*p:].reshape(p+1, len(HEADS))
    y[:, -1] = y[:, -1].clip(0., 1.)
    return y


def accept(y):
    return (y[:, 0] > 0.) & (y[:, 1] <= 0.) & (y[:, 2] <= 0.) & (y[:, 3] >= .5)
