"""Descriptive model-boundary attribution; never a deployment certificate."""
import numpy as np


def summarize(target, old, new, moments, weights, target_scale):
    target, old, new, moments, weights, target_scale = map(np.asarray,
        (target, old, new, moments, weights, target_scale))
    n = len(target)
    if (target.shape != (n, 2) or old.shape != target.shape or new.shape != target.shape
            or moments.shape != (n, 5) or weights.shape != (n,) or target_scale.shape != (2,)
            or not all(np.isfinite(a).all() for a in (target, old, new, moments, weights, target_scale))
            or (weights < 0).any() or weights.sum() <= 0 or (target_scale <= 0).any()):
        raise ValueError('Aligned known supervision, predictions and fitting-only scales required')
    w = weights/weights.sum()
    safe = (target <= 0).all(1)
    a, b = (old <= 0).all(1), (new <= 0).all(1)
    groups = {'both_reject': ~a & ~b, 'both_accept': a & b,
        'new_only_accept': ~a & b, 'old_only_accept': a & ~b}
    error_change = (((old-target)/target_scale)**2-((new-target)/target_scale)**2).mean(1)
    strata = {}
    for name, at in groups.items():
        mass = float(w[at].sum())
        strata[name] = dict(rows=int(at.sum()), weight=mass,
            weighted_MSE_reduction_contribution=float(w[at]@error_change[at]),
            known_joint_safe=int((at & safe).sum()), known_easy=int((at & (moments[:, 3] > 0)).sum()))
    decisions = {}
    for name, prediction in (('old', old), ('new', new)):
        all_ok, easy_ok = (prediction <= 0).T
        reasons = {'accept': all_ok & easy_ok, 'all_only': ~all_ok & easy_ok,
            'easy_only': all_ok & ~easy_ok, 'both': ~all_ok & ~easy_ok}
        decisions[name] = {}
        for reason, at in reasons.items():
            m = (w[at, None]*moments[at]).sum(0)
            decisions[name][reason] = dict(rows=int(at.sum()), weight=float(w[at].sum()),
                true_joint_safe_rows=int((at & safe).sum()),
                true_joint_unsafe_rows=int((at & ~safe).sum()),
                easy_reference_mass=float(m[3]), all_reference_mass=float(m[2]),
                positive_harm_mass=float(m[1]), easy_positive_harm_mass=float(m[4]),
                selected_all_risk=float(m[1]/m[2]) if m[2] > 0 else None,
                selected_easy_risk=float(m[4]/m[3]) if m[3] > 0 else None)
    return dict(strata=strata, decisions=decisions,
        weighted_MSE_reduction=float(w@error_change),
        non_easy_target_zero_rows=int((moments[:, 3] == 0).sum()),
        mixed_sign_rows=int(((target.min(1) < 0) & (target.max(1) > 0)).sum()),
        known_only=True, selected_ratios_not_full_policy=True,
        population_safety_certificate=False, support_gate_not_applied=True)
