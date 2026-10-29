"""Label-aware geometric relaxation, not a realizable model or safety policy."""
import numpy as np


def bound_relaxation(baseline, budget, cv, target, valid):
    baseline, budget, cv, target = map(lambda x: np.asarray(x, float),
                                     (baseline, budget, cv, target))
    valid = np.asarray(valid)
    if (baseline.ndim != 3 or baseline.shape[-1] != 2 or cv.shape != baseline.shape
            or target.shape != baseline.shape or budget.shape != baseline.shape[:2]
            or valid.shape != budget.shape or valid.dtype != bool
            or not np.isfinite(baseline).all() or not np.isfinite(cv).all()
            or not np.isfinite(budget).all() or (budget < 0).any()
            or not np.isfinite(target[valid]).all()):
        raise ValueError('Finite causal trajectories/budgets and explicitly masked labels required')
    margin = budget - np.linalg.norm(cv - baseline, axis=-1)
    tolerance = 1e-6 * np.maximum(1., budget)
    strictly_inside = margin > tolerance
    exact_fixed = (budget == 0) & (np.linalg.norm(cv-baseline, axis=-1) == 0)
    finite_cv = (strictly_inside | exact_fixed).all(1)
    boundary = ((~strictly_inside & ~exact_fixed) & (np.abs(margin) <= tolerance)).any(1)
    outside = (margin < -tolerance).any(1)
    safe = np.where(valid[..., None], target, 0.)
    error_infimum = np.maximum(np.linalg.norm(safe-baseline, axis=-1)-budget, 0.)
    n = valid.sum(1)
    lower_ade = np.divide(np.where(valid, error_infimum, 0.).sum(1), n,
                          out=np.full(len(n), np.nan), where=n > 0)
    return dict(finite_CV_geometrically_representable=finite_cv,
                CV_boundary_or_numerically_ambiguous=boundary & ~outside,
                CV_outside_bound=outside, minimum_CV_margin=margin.min(1),
                independent_step_ball_ADE_infimum=lower_ade,
                exactly_zero_motion_budget=(budget == 0).all(1))
