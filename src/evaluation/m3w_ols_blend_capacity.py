"""Nested OLS4-centered proposal menus for the exact-count capacity audit."""
import numpy as np

FRACTIONS = (.25, .50, .75)
MIN_NEW_CHANGE = 1e-6


def add_blends(cv, ols, bank, source):
    """Retain all original options; add nontrivial paths using only forecasts."""
    options = []
    for j in source:
        for f in FRACTIONS:
            new = ols+f*(bank[:, j]-ols)
            close = np.linalg.norm(new-cv, axis=-1).max(1) <= MIN_NEW_CHANGE
            new[close] = cv[close]
            options.append(new)
    return np.concatenate((bank, np.stack(options, axis=1)), axis=1)


def prune_unknown(cv, ols, bank, valid):
    """Prune same-person dual-radius dominance, never dominance by abstention."""
    known = valid.any(1)
    if known.all():
        return bank
    rows = np.flatnonzero(~known)
    p = bank[rows]
    cv_dist = np.linalg.norm(p-cv[rows, None], axis=-1).max(2)
    ols_dist = np.linalg.norm(p-ols[rows, None], axis=-1).max(2)
    legal = cv_dist > 0.
    keep = legal.copy()
    for j in range(p.shape[1]):
        le = (cv_dist <= cv_dist[:, j, None]) & (ols_dist <= ols_dist[:, j, None])
        strict = (cv_dist < cv_dist[:, j, None]) | (ols_dist < ols_dist[:, j, None])
        earlier = np.arange(p.shape[1]) < j
        keep[:, j] &= ~(legal & le & (strict | earlier)).any(1)
    pruned = bank.copy()
    for j in range(p.shape[1]):
        replace = rows[~keep[:, j]]
        pruned[replace, j] = cv[replace]
    return pruned
