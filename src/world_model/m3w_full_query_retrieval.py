"""Learn recipient opportunity from full-query, past-only candidate features."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model.m3w_native_support_retrieval import features as history_features


SHORTLIST = 8
RIDGE = .02


def opportunity(cv, bank, y, valid, scale, groups, easy_cut, cap):
    """Fitting labels: best locally feasible gain involving each recipient."""
    ref, _ = errors(cv, y, valid)
    known = valid.any(1)
    easy = known & (ref > 0.) & (ref <= easy_cut)
    gains = []
    for path in bank.transpose(1, 0, 2, 3):
        score, _ = errors(path, y, valid)
        radius = np.linalg.norm(path-cv, axis=-1).max(1)
        gains.append(np.where(radius > 0., np.where(known, ref-score, -radius), -np.inf))
    gain = np.stack(gains, axis=1).max(1)
    safe = np.zeros(len(cv))
    usable = known & np.isfinite(gain)
    r = np.where(known, ref, 0.)
    g = np.where(usable, gain, 0.)
    harm = np.maximum(-g, 0.)
    all_cost = harm - .02*r
    easy_cost = np.where(easy, all_cost, 0.)
    easy_net = np.where(easy, -g, 0.)
    for q, k in zip(groups, cap, strict=True):
        if k == 0 or not (easy[q] & usable[q]).any():
            continue
        i = q[usable[q]]
        bound = .02*r[q[easy[q]]].sum()
        solo = easy[i] & (g[i] > 0.) & (all_cost[i] <= 0.) & (easy_cost[i] <= 0.) & (easy_net[i] <= bound)
        safe[i] = np.where(solo, g[i]/scale[i], 0.)
        if k == 2:
            value = g[i, None]+g[i]
            feasible = ((easy[i, None] | easy[i]) & (value > 0.)
                        & (all_cost[i, None]+all_cost[i] <= 0.)
                        & (easy_cost[i, None]+easy_cost[i] <= 0.)
                        & (easy_net[i, None]+easy_net[i] <= bound))
            np.fill_diagonal(feasible, False)
            # Half of the normalized pair gain is credited to each member.
            relative = .5*(g[i, None]/scale[i, None]+g[i]/scale[i])
            safe[i] = np.maximum(safe[i], np.where(feasible, relative, 0.).max(1))
    return np.column_stack((np.where(np.isfinite(gain), gain/scale, 0.), safe)).clip(-8., 8.)


def features(history, bank, cv, ols, groups, crowd):
    past = history_features(history, bank, cv, crowd, groups)
    d = np.linalg.norm(bank-ols[:, None], axis=-1)
    c = np.linalg.norm(bank-cv[:, None], axis=-1)
    v = np.column_stack((past, np.log1p(d[:, :, -1]), np.log1p(c[:, :, -1]),
                         np.log1p(np.linalg.norm(ols-cv, axis=-1).mean(1))))
    return np.column_stack((np.ones(len(v)), v, np.tanh(v)))


def moments(x, y, weight):
    w = weight/weight.sum()
    return x.T@(w[:, None]*x), x.T@(w[:, None]*y)


def coefficients(parts):
    gram, rhs = (np.mean([p[k] for p in parts], axis=0) for k in (0, 1))
    reg = np.eye(len(gram))*RIDGE
    reg[0, 0] = 0.
    return np.linalg.solve(gram+reg, rhs)


def lists(scores, legal):
    eligible = np.flatnonzero(legal.any(1))
    gain = eligible[np.argsort(-scores[eligible, 0], kind='stable')]
    size = min(SHORTLIST, len(gain))
    cut = (size+1)//2
    reserve = gain[:cut]
    remainder = gain[cut:]
    safe = remainder[np.argsort(-scores[remainder, 1], kind='stable')]
    return gain[:size], np.r_[reserve, safe[:size-cut]]
