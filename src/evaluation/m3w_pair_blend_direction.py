"""Convex directional test for pair-relative improvement in a fixed affine slate."""
import numpy as np
from scipy.optimize import minimize_scalar

TOL = 1e-9


def slope_terms(residual, coeff, observed):
    """Split the one-sided directional derivative into linear and norm terms."""
    norm = np.linalg.norm(residual, axis=1)
    smooth = observed & (norm > 0.)
    linear = np.einsum('ti,tik->k', residual[smooth]/norm[smooth, None], coeff[smooth])
    nonlinear = coeff[~smooth]
    # With nonnegative alpha, single-endpoint terms are already linear.
    active = (nonlinear != 0.).any(1)
    one = active.sum(1) <= 1
    linear += np.linalg.norm(nonlinear[one], axis=1).sum(0)
    return linear, nonlinear[~one]


def minimum_slope(linear, nonlinear):
    """Bound the minimum directional derivative over the unit simplex, k <= 2."""
    k = len(linear)
    if k not in (1, 2):
        raise ValueError('The fixed query budget allows only one or two recipients')
    if not len(nonlinear):
        direction = np.eye(k)[np.argmin(linear)]
        return float(linear.min()), float(linear.min()), direction
    p, q = nonlinear[..., 1], nonlinear[..., 0]-nonlinear[..., 1]
    tilt = linear[0]-linear[1]
    lo, hi = 0., 1.
    best, at = np.inf, 0.
    lower = -np.inf
    for _ in range(54):
        mid = (lo+hi)/2.
        v = p+mid*q
        lengths = np.linalg.norm(v, axis=1)
        score = float(linear[1]+mid*tilt+lengths.sum())
        nonzero = lengths > 0.
        slope = float(tilt+np.sum(np.sum(v[nonzero]*q[nonzero], axis=1)/lengths[nonzero]))
        if score < best:
            best, at = score, mid
        lower = max(lower, score+min(slope*(lo-mid), slope*(hi-mid)))
        if slope > 0.:
            hi = mid
        elif slope < 0.:
            lo = mid
        else:
            break
    return min(lower, best), best, np.array([at, 1.-at])


def evaluate(history, reference, proposal, cv, target, valid, switched, edge_fn):
    active = np.flatnonzero(switched)
    if len(active) not in (1, 2):
        raise ValueError('Directional test requires a nonempty original query budget')
    i, j = edge_fn(history)
    clean = np.where(valid[..., None], target, 0.)
    observed = (valid[i] & valid[j]).ravel()
    residual = (reference[i]-reference[j]-clean[i]+clean[j]).reshape(-1, 2)
    coeff = np.zeros((len(i), valid.shape[1], 2, len(active)))
    delta = proposal[active]-reference[active]
    for col, row in enumerate(active):
        coeff[i == row, :, :, col] += delta[col]
        coeff[j == row, :, :, col] -= delta[col]
    full_steps = len(observed)
    if not full_steps:
        raise ValueError('An active query must have at least one requested edge')
    coeff = coeff.reshape(-1, 2, len(active))
    affected = (coeff != 0.).any((1, 2))
    residual, coeff, observed = residual[affected], coeff[affected], observed[affected]
    linear, nonlinear = slope_terms(residual, coeff, observed)
    known_norm = np.linalg.norm(residual[observed], axis=1)
    lo, hi, best = minimum_slope(linear, nonlinear)
    equal = np.full(len(active), 1./len(active))
    shared_slope = float(linear@equal+np.linalg.norm(nonlinear@equal, axis=1).sum())

    def solve(direction, lower, upper):
        if lower/full_steps >= -TOL:
            return 'no_gain_to_tolerance', 0., reference
        if upper/full_steps >= -TOL:
            return 'unresolved', 0., reference
        move = coeff@direction
        extent = 1./direction.max()

        def loss(t):
            diff = t*move
            a = np.linalg.norm(residual[observed]+diff[observed], axis=1)-known_norm
            return float((a.sum()+np.linalg.norm(diff[~observed], axis=1).sum())/full_steps)

        fit = minimize_scalar(loss, bounds=(0., extent), method='bounded', options={'xatol': 1e-10})
        step = (fit.x, extent, *(extent*2.**-k for k in range(1, 21)))
        for t in sorted(step, key=loss):
            gain = -loss(t)
            if gain <= TOL:
                break
            forecast = reference.copy()
            forecast[active] += (t*direction)[:, None, None]*delta
            if np.all(np.linalg.norm(forecast[active]-cv[active], axis=2).max(1) > 1e-6):
                return 'verified_gain', gain, forecast
        return 'unresolved', 0., reference

    shared = solve(equal, shared_slope, shared_slope)
    free = solve(best, lo, hi)
    if shared[0] == 'verified_gain' and free[0] == 'no_gain_to_tolerance':
        raise ValueError('Joint direction bound contradicts a shared feasible witness')
    if shared[1] > free[1]:
        free = shared
    return shared, free
