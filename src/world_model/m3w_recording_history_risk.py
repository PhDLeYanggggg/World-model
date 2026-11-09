"""Observed-prefix risk proxies and small source-weighted calibration models."""

import numpy as np

from src.world_model import m3w_dronecrowd_motion_floor as floor


SIGNALS = ("donor", "history", "combined")
PENALTY = .01


def prefix_masses(history, choices, changed_from_cv, easy_cut):
    """Score frozen action identities on four observed positions, not future labels."""
    past = floor.paths(floor.base(history[:, :4]), horizon=4)
    observed = (history[:, 4:]-history[:, 3:4])[:, None]
    e = np.linalg.norm(past-observed, axis=-1).mean(2)
    cv = e[:, 0]
    # Rescale only the proxy cutoff by the mean index of each forecast span.
    easy = (cv > 0.) & (cv <= easy_cut*5./13.)
    row = np.arange(len(e))[:, None, None]
    changed = changed_from_cv[row, choices]
    h = np.where(changed, np.maximum(e[row, choices+1]-cv[:, None, None], 0.), 0.)
    ref = np.where(changed, cv[:, None, None], 0.)
    mask = easy[:, None, None]
    return np.stack((h.sum(0), (h*mask).sum(0), ref.sum(0), (ref*mask).sum(0)), axis=-1)


def fit_predict(x, y, groups, fit, pred):
    """Ridge on log-scaled risk, per arm and endpoint, with equal source weights."""
    z = np.log1p(100.*np.asarray(x))
    label = np.log1p(100.*np.asarray(y))
    groups = np.asarray(groups)
    if not len(fit) or set(groups[fit]) & set(groups[pred]):
        raise ValueError("Calibration and prediction must be source-disjoint")
    g, inv, count = np.unique(groups[fit], return_inverse=True, return_counts=True)
    w = 1./(count[inv]*len(g))
    result = np.zeros_like(label[pred])
    for i in range(z.shape[1]):
        for j in range(z.shape[2]):
            a = z[fit, i, j]
            b = label[fit, i, j]
            mu, avg = w@a, w@b
            d = a-mu
            coef = np.linalg.solve(d.T@(w[:, None]*d)+PENALTY*np.eye(a.shape[1]), d.T@(w*(b-avg)))
            value = (z[pred, i, j]-mu)@coef+avg
            result[:, i, j] = np.expm1(np.clip(value, 0., 40.))/100.
    return result


def source_folds(x, y, groups):
    groups = np.asarray(groups)
    point, upper = np.zeros_like(y), np.zeros_like(y)
    for g in np.unique(groups):
        outer, held = np.flatnonzero(groups != g), np.flatnonzero(groups == g)
        value = fit_predict(x, y, groups, outer, held)
        residuals = []
        for j in np.unique(groups[outer]):
            train, test = outer[groups[outer] != j], outer[groups[outer] == j]
            residuals.append((y[test]-fit_predict(x, y, groups, train, test)).max(0))
        point[held] = value
        upper[held] = value+np.maximum(np.max(residuals, axis=0), 0.)
    return point, upper
