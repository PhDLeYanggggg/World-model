"""Exact infeasibility pruning before ratio-based risk optimization.

This does not alter the already frozen European conditional-risk V1 readout.
"""
import numpy as np


def bounded_risk_coefficients(utility, moments, moving, *, budget, support_available):
    u, m, moving = np.asarray(utility, float), np.asarray(moments, float), np.asarray(moving)
    if (u.ndim != 1 or not len(u) or m.shape != (len(u), 2) or moving.shape != u.shape
            or moving.dtype != bool or not np.isfinite(u).all() or not np.isfinite(m).all()
            or np.any(m < 0) or not np.isfinite(budget) or budget < 0
            or not isinstance(support_available, (bool, np.bool_))):
        raise ValueError('Finite nonnegative event moments and causal support required')
    total = float(m[:, 0].sum())
    if not np.isfinite(total):
        raise ValueError('Nonfinite predicted event mass')
    cap = budget*total
    if not np.isfinite(cap):
        raise ValueError('Nonfinite predicted risk cap')
    # With nonnegative costs, one cost above the whole budget is infeasible
    # in every subset. Prune it before division, not by enlarging epsilon.
    eligible = moving & (u > 0) & support_available & (total > 0) & (m[:, 1] <= cap)
    risk = np.zeros(len(u))
    if eligible.any():
        risk[eligible] = m[eligible, 1]/(total/len(u))
    if not np.isfinite(risk).all():
        raise ValueError('Nonfinite bounded risk coefficient')
    return dict(expected_gain=np.maximum(u, 0), expected_harm=risk,
                supported=eligible, predicted_mass=total, native_budget=cap)
