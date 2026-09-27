"""Small monotone harm readouts with optional fitting-only moment constraints."""
from itertools import combinations
import numpy as np
from src.world_model.m3w_cost_mass import fitting_weights

KNOTS = np.array([0., .1, .3, .6, 1.])


def inputs(prediction, envelope):
    p, e = np.asarray(prediction, dtype=float), np.asarray(envelope, dtype=float)
    if (p.shape != (len(e), 4) or e.ndim != 1 or not np.isfinite(p).all()
            or not np.isfinite(e).all() or (p < 0).any() or (e < 0).any()
            or np.any(p[:, 1] > e + 1e-8) or np.any(p[:, 3] > p[:, 1] + 1e-8)):
        raise ValueError('Finite aligned nested causal costs required')
    all_fraction = np.divide(p[:, 1], e, out=np.zeros(len(e)), where=e > 0)
    easy_fraction = np.divide(p[:, 3], p[:, 1], out=np.zeros(len(e)), where=p[:, 1] > 0)
    return p, e, np.clip(all_fraction, 0, 1), np.clip(easy_fraction, 0, 1)


def basis(fraction, cap):
    # Nonnegative simplex increments encode increasing knot ordinates in [0,1].
    hat = np.column_stack([np.interp(fraction, KNOTS, row) for row in np.eye(len(KNOTS))])
    tail = np.cumsum(hat[:, ::-1], axis=1)[:, ::-1]
    return np.column_stack([cap[:, None] * tail, np.zeros(len(cap))])


def vertices(mean, target):
    """All vertices of a simplex intersected with one moment hyperplane."""
    n = len(mean)
    if target is None:
        return np.eye(n)
    result = []
    for i in range(n):
        if abs(mean[i] - target) <= 1e-12:
            result.append(np.eye(n)[i])
    for i, j in combinations(range(n), 2):
        difference = mean[i] - mean[j]
        if abs(difference) < 1e-14:
            continue
        weight = (target - mean[j]) / difference
        if 0 <= weight <= 1:
            v = np.zeros(n); v[i], v[j] = weight, 1 - weight; result.append(v)
    if not result:
        raise ValueError('Infeasible fitting moment')
    return np.asarray(result)


def component(fraction, cap, target, weights, preserve_mass):
    a = basis(fraction, cap); w, y = np.asarray(weights), np.asarray(target)
    capacity, goal = float(w @ cap), float(w @ y)
    if goal < -1e-12 or goal > capacity + 1e-8 * (1 + capacity):
        if preserve_mass:
            raise ValueError('Target mean exceeds nested causal capacity')
    if capacity == 0:
        return dict(ordinates=[0.] * len(KNOTS), MSE=float(w @ y**2),
            target_mass=goal, predicted_mass=0., normalized_optimality_gap=0.,
            mass_preserved=bool(goal == 0), positive_increments=0)
    scale = max(float(w @ cap**2), 1e-20)
    q, b = (a.T @ (w[:, None] * a)) / scale, (a.T @ (w*y)) / scale
    mean = (w @ a) / capacity
    level = float(np.clip(goal / capacity, 0, 1)) if preserve_mass else None
    v = vertices(mean, level)
    e = np.vstack([np.ones(a.shape[1]), mean]) if preserve_mass else np.ones((1, a.shape[1]))
    rhs = np.array([1., level]) if preserve_mass else np.ones(1)
    candidates = [row for row in v]
    # Enumerating 63 supports avoids an iterative optimizer or tuning a penalty.
    for size in range(1, a.shape[1] + 1):
        for support in combinations(range(a.shape[1]), size):
            indices = list(support); es = e[:, indices]
            kkt = np.block([[q[np.ix_(indices, indices)], es.T],
                            [es, np.zeros((len(rhs), len(rhs)))]])
            vector = np.r_[b[indices], rhs]
            solution = np.linalg.lstsq(kkt, vector, rcond=1e-12)[0]
            if np.max(np.abs(kkt @ solution - vector)) > 1e-8:
                continue
            d = np.zeros(a.shape[1]); d[indices] = solution[:size]
            if d.min() < -1e-10:
                continue
            d = np.maximum(d, 0); d /= d.sum()
            if np.max(np.abs(e @ d - rhs)) <= 1e-9:
                candidates.append(d)
    d = min(candidates, key=lambda z: float(z @ q @ z - 2*b @ z))
    gradient = 2 * (q @ d - b)
    gap = float(gradient @ d - np.min(v @ gradient))
    if gap > 1e-7 or np.max(np.abs(e @ d - rhs)) > 1e-9:
        raise RuntimeError('Convex optimality or feasibility certificate failed')
    ordinates = np.cumsum(d[:-1]); prediction = cap * np.interp(fraction, KNOTS, ordinates)
    achieved = float(w @ prediction)
    matched = abs(achieved-goal) <= 1e-8 * (1+goal)
    if preserve_mass and not matched:
        raise RuntimeError('Fitting moment was not preserved')
    return dict(ordinates=ordinates.tolist(), MSE=float(w @ (prediction-y)**2),
        target_mass=goal, predicted_mass=achieved, normalized_optimality_gap=max(0.,gap),
        mass_preserved=bool(matched), positive_increments=int((d > 1e-10).sum()))


def predict(model, prediction, envelope):
    p, e, r, u = inputs(prediction, envelope)
    if model['knots'] != KNOTS.tolist():
        raise ValueError('Frozen knot schema mismatch')
    out = p.copy()
    for j, (fraction, cap) in enumerate(((r, e), (u, None))):
        ordinates = np.asarray(model['components'][j]['ordinates'])
        if (ordinates.shape != KNOTS.shape or not np.isfinite(ordinates).all()
                or ordinates.min() < -1e-10 or ordinates.max() > 1+1e-10
                or (np.diff(ordinates) < -1e-10).any()):
            raise ValueError('Invalid monotone readout')
        out[:, (1,3)[j]] = (e if j == 0 else out[:,1]) * np.interp(fraction, KNOTS, ordinates)
    return out


def fit(prediction, target, envelope, sites, outer, *, preserve_mass):
    p, e, r, u = inputs(prediction, envelope)
    y = np.asarray(target, dtype=float)
    w = fitting_weights(p, y, e, sites, outer); take = w > 0
    if (np.any(y[take,3] > y[take,1] + 1e-8) or np.any(y[take,1] > e[take] + 1e-8)):
        raise ValueError('Target harm must obey the causal envelope')
    a = component(r[take],e[take],y[take,1],w[take],preserve_mass)
    cap = e[take] * np.interp(r[take],KNOTS,a['ordinates'])
    b = component(u[take],cap,y[take,3],w[take],preserve_mass)
    return dict(knots=KNOTS.tolist(), components=[a,b], preserve_mass=preserve_mass,
        fitting_rows=int(take.sum()),training_sites=sorted(set(sites)),
        mass_preserved=all(c['mass_preserved'] for c in (a,b)),
        calibration_guarantee=False, objective='sequential_equal_locality_monotone_MSE')
