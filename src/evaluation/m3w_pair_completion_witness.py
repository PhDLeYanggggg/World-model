"""Construct coherent label completions for an exposed pair-score sensitivity."""
import numpy as np
from scipy.optimize import minimize


def completion_ledger(agents, frames, history, target, valid):
    """Map one recording to a unique agent/frame coordinate ledger."""
    h, t = history.shape[1], target.shape[1]
    start = int(frames.min())-h+1
    length = int(frames.max())+t-start+1
    _, track = np.unique(agents, return_inverse=True)
    here = track*length+frames-start
    hx = here[:, None]+np.arange(1-h, 1)
    tx = here[:, None]+np.arange(1, t+1)
    index = np.r_[hx.ravel(), tx[valid]]
    xy = np.concatenate((history.reshape(-1, 2), target[valid]))
    if not np.isfinite(xy).all():
        raise ValueError('Require finite released coordinates')
    order = np.argsort(index, kind='stable')
    a, b = order[:-1], order[1:]
    repeat = index[a] == index[b]
    if not np.array_equal(xy[a[repeat]], xy[b[repeat]]):
        raise ValueError('Released history/target coordinates conflict across windows')
    n = (int(track.max())+1)*length
    known, base = np.zeros(n, dtype=bool), np.zeros((n, 2))
    known[index], base[index] = True, xy
    if not np.array_equal(known[tx], valid):
        raise ValueError('A released future mask conflicts with another observation')
    radius, gaps = np.zeros(n), np.full(n, -1, dtype=np.int64)
    missing = np.unique(tx[~valid])
    for agent in np.unique(missing//length):
        start = agent*length
        obs = np.flatnonzero(known[start:start+length])
        ix = missing[missing//length == agent]
        requested = ix-start
        coords = base[start+obs]
        base[ix] = np.column_stack([np.interp(requested, obs, z) for z in coords.T])
        insert = np.searchsorted(obs, requested)
        left = obs[np.maximum(0, insert-1)]
        right = obs[np.minimum(insert, len(obs)-1)]
        distance = np.minimum(np.abs(requested-left), np.abs(requested-right))
        speeds = np.linalg.norm(np.diff(coords, axis=0)[np.diff(obs) == 1], axis=1)
        scale = max(.25, float(np.median(speeds))) if len(speeds) else .25
        radius[ix] = scale*np.minimum(distance, t)
        gaps[ix] = agent*(length+1)+insert
    return tx, known, base, radius, gaps


def extreme(base, radius, gaps, edges, a, b, weight, direction):
    """Find a feasible witness, not a global extremum or a label imputation."""
    if direction not in (-1., 1.) or edges.shape != (len(weight), 2) or np.any(weight <= 0.):
        raise ValueError('Expected weighted pair terms and a fixed objective direction')
    if not len(edges):
        return 0., base, dict(groups=0, selected_start=None, solver_success=None)
    left, right = edges.T
    r = radius[edges]
    groups = np.unique(gaps[edges][r > 0.])
    if not len(groups) or (groups < 0).any():
        raise ValueError('Only pair terms with a genuinely missing endpoint enter search')
    g = np.where(r > 0., np.searchsorted(groups, gaps[edges]), len(groups))
    base_rel = base[left]-base[right]
    factor = direction/weight.sum()

    def calculate(flat):
        u = np.vstack((flat.reshape(-1, 2), np.zeros((1, 2))))
        diff = base_rel+r[:, :1]*u[g[:, 0]]-r[:, 1:]*u[g[:, 1]]
        za, zb = diff-a, diff-b
        da, db = np.linalg.norm(za, axis=1), np.linalg.norm(zb, axis=1)
        qa = np.divide(za, da[:, None], out=np.zeros_like(za), where=da[:, None] > 0.)
        qb = np.divide(zb, db[:, None], out=np.zeros_like(zb), where=db[:, None] > 0.)
        grad = (qa-qb)*weight[:, None]
        grouped = np.zeros_like(u)
        np.add.at(grouped, g[:, 0], grad*r[:, :1])
        np.add.at(grouped, g[:, 1], -grad*r[:, 1:])
        return -factor*float(weight@(da-db)), -factor*grouped[:-1].ravel()

    zero = np.zeros(2*len(groups))
    fixed = np.random.default_rng(9419).uniform(-.5, .5, size=zero.shape)
    best = (calculate(zero)[0], zero, None, None)
    for i, start in enumerate((zero, fixed)):
        fit = minimize(calculate, start, jac=True, method='L-BFGS-B', bounds=[(-1., 1.)]*len(zero),
                       options=dict(maxiter=90, maxls=40, ftol=1e-11, gtol=1e-7))
        if np.isfinite(fit.x).all() and (np.abs(fit.x) <= 1.).all():
            loss = calculate(fit.x)[0]
            if loss < best[0]:
                best = (loss, fit.x, i, bool(fit.success))
    loss, solution, start, success = best
    out = base.copy()
    use = (radius > 0.) & np.isin(gaps, groups)
    out[use] += radius[use, None]*solution.reshape(-1, 2)[np.searchsorted(groups, gaps[use])]
    return -loss/factor, out, dict(groups=len(groups), selected_start=start, solver_success=success)
