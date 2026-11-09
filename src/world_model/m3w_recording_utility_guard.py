"""Observed-prefix utility and source-held gain calibration for fixed paths."""

import numpy as np

from src.world_model import m3w_dronecrowd_motion_floor as floor


PENALTY = .1
SIGNALS = ("constant", "donor", "history", "combined")


def prefix_utility(history, choices):
    """One query: past-only OLS4-minus-action error and CV normalization mass."""
    bank = floor.paths(floor.base(history[:, :4]), horizon=4)
    past = (history[:, 4:]-history[:, 3:4])[:, None]
    err = np.linalg.norm(bank-past, axis=-1).mean(2)
    use = err[np.arange(len(history))[:, None], choices+1]
    return (err[:, 1:2]-use).mean(0), err[:, 0].mean()


def fit_predict(x, y, groups, fit, pred):
    if not len(fit) or set(groups[fit]) & set(groups[pred]):
        raise ValueError("Fit and prediction must be source-disjoint")
    g, inv, count = np.unique(groups[fit], return_inverse=True, return_counts=True)
    w = 1./(count[inv]*len(g))
    avg = w@y[fit]
    if not x.shape[1]:
        return np.full(len(pred), avg)
    center = w@x[fit]
    spread = np.sqrt(w@((x[fit]-center)**2)).clip(1e-6)
    a = (x[fit]-center)/spread
    coef = np.linalg.solve(a.T@(w[:, None]*a)+PENALTY*np.eye(a.shape[1]), a.T@(w*(y[fit]-avg)))
    return (x[pred]-center)/spread@coef+avg


def folds(x, y, groups):
    point, lower = np.zeros_like(y), np.zeros_like(y)
    for g in np.unique(groups):
        train, held = np.flatnonzero(groups != g), np.flatnonzero(groups == g)
        point[held] = fit_predict(x, y, groups, train, held)
        gaps = [0.]
        for h in np.unique(groups[train]):
            fit, val = train[groups[train] != h], train[groups[train] == h]
            gaps.append(float((fit_predict(x, y, groups, fit, val)-y[val]).max()))
        lower[held] = point[held]-max(gaps)
    return point, lower
