"""Fitting-only moment matching under the existing nested causal envelope."""
import numpy as np
from src.world_model.m3w_oof_magnitude import predict_magnitude as predict


def fitting_weights(prediction, target, envelope, sites, outer):
    p, y, env, sites = map(np.asarray, (prediction, target, envelope, sites))
    if (p.shape != y.shape or p.shape != (len(env), 4) or sites.shape != env.shape
            or len(p) == 0 or outer in sites or not np.isfinite(p).all()
            or not np.isfinite(env).all() or (p < 0).any() or (env < 0).any()):
        raise ValueError('Finite, aligned fitting-only cost predictions required')
    known = np.isfinite(y).all(1)
    if not np.array_equal(np.isnan(y).all(1), ~known) or (y[known] < 0).any():
        raise ValueError('Unknown labels must be paired; finite labels nonnegative')
    valid = known & (env > 0)
    names = sorted(set(sites)); w = np.zeros(len(p))
    for site in names:
        use = valid & (sites == site)
        if not use.any():
            raise ValueError('Every fitting locality needs positive-envelope support')
        w[use] = 1/(len(names)*use.sum())
    return w


def _root(p, cap, w, target, bound, iterations):
    capacity = float(w @ np.minimum(cap, bound*p))
    tol = 1e-12*(1+target)
    if target == 0:
        slope, status = 0., 'zero_target'
    elif target > capacity+tol:
        slope, status = bound, 'infeasible_at_bound'
    else:
        lo, hi = 0., bound
        for _ in range(iterations):
            middle = (lo+hi)/2
            if float(w @ np.minimum(cap, middle*p)) < target:
                lo = middle
            else:
                hi = middle
        slope, status = (lo+hi)/2, 'matched'
    achieved = float(w @ np.minimum(cap, slope*p))
    return float(slope), dict(status=status, target_mass=float(target), achieved_mass=achieved,
        maximum_mass=capacity, error=achieved-target,
        within_numeric_tolerance=abs(achieved-target) <= tol)


def fit(prediction, target, envelope, sites, outer, *, max_slope=8., iterations=80):
    if not np.isfinite(max_slope) or max_slope <= 0 or iterations != 80:
        raise ValueError('Positive slope bound and fixed 80-step root solve required')
    p, y, env, sites = map(np.asarray, (prediction, target, envelope, sites))
    w = fitting_weights(p, y, env, sites, outer); use = w > 0
    a, b, e, q = p[use], y[use], env[use], w[use]
    first, c1 = _root(a[:, 1], e, q, float(q @ b[:, 1]), max_slope, iterations)
    cap = np.minimum(e, first*a[:, 1])
    second, c2 = _root(a[:, 3], cap, q, float(q @ b[:, 3]), max_slope, iterations)
    return dict(slopes=[first, second], max_slope=max_slope, iterations=iterations,
        components=[c1, c2], mass_preserved=all(c['within_numeric_tolerance'] for c in (c1, c2)),
        fitting_rows=int(use.sum()), training_sites=sorted(set(sites)),
        objective='equal_locality_projected_moment_matching_not_MSE_minimization',
        calibration_guarantee=False)


def decompose(prediction, target, scored, weights, sites, model):
    """Describe the same squared loss by target support and fitting locality."""
    p, y, scored, weights, sites = map(np.asarray, (prediction, target, scored, weights, sites))
    use = weights > 0
    if (p.shape != y.shape or scored.shape != p.shape or len(weights) != len(p)
            or (weights < 0).any() or not np.isclose(weights.sum(), 1.)
            or not np.isfinite(y[use]).all() or not np.isfinite(scored[use]).all()):
        raise ValueError('Supported normalized diagnostic weights required')
    p, y, scored, w, names = p[use], y[use], scored[use], weights[use], sites[use]
    out = {}
    for j, (col, label) in enumerate(((1, 'H_all'), (3, 'H_easy'))):
        x, truth, got = p[:, col], y[:, col], scored[:, col]
        zero = truth == 0; error = (got-truth)**2
        denominator = float(w @ x**2); numerator = float(w @ (x*truth))
        target_mass, raw_mass, scored_mass = [float(w @ v) for v in (truth, x, got)]
        local = {}
        for site in sorted(set(names)):
            s = names == site
            local[str(site)] = dict(rows=int(s.sum()), weight_mass=float(w[s].sum()),
                MSE_contribution=float(w[s] @ error[s]),
                zero_target_SSE=float(w[s & zero] @ error[s & zero]),
                positive_target_SSE=float(w[s & ~zero] @ error[s & ~zero]),
                target_mass_contribution=float(w[s] @ truth[s]),
                predicted_mass_contribution=float(w[s] @ got[s]))
        out[label] = dict(rows=len(w), positive_rows=int((~zero).sum()),
            zero_target_weight=float(w[zero].sum()), MSE=float(w @ error),
            zero_target_SSE=float(w[zero] @ error[zero]),
            positive_target_SSE=float(w[~zero] @ error[~zero]),
            target_mass=target_mass, raw_predicted_mass=raw_mass, predicted_mass=scored_mass,
            mass_ratio=scored_mass/target_mass if target_mass > 0 else None,
            LS_numerator=numerator, LS_denominator=denominator,
            LS_denominator_zero_share=float(w[zero] @ x[zero]**2)/denominator if denominator > 0 else None,
            unprojected_LS_slope=numerator/denominator if denominator > 0 else None,
            unprojected_mean_slope=target_mass/raw_mass if raw_mass > 0 else None,
            projection_mass_loss=float(w @ (model['slopes'][j]*x-got)), localities=local)
    return out
