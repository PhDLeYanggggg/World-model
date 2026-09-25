"""Aligned gain and two-event harm learning between fixed causal policies."""
import numpy as np
from src.world_model.m3w_european_source_intervention import causal_cost_features
from src.world_model.m3w_floor_relative import relative_targets
from src.world_model.m3w_geometric_cost_head import rollout_envelope

POLICIES = ('reference_easy', 'candidate_all', 'utility_only', 'all_risk_only',
            'easy_risk_only', 'dual_risk', 'ridge_dual')


def features(geometry, cv, reference, candidate, bits):
    """Bits describe frozen policies, never future support or event membership."""
    bits = np.asarray(bits)
    if bits.dtype != bool or bits.shape != (len(cv), 4):
        raise ValueError('Four aligned causal policy bits required')
    x, scale = causal_cost_features(geometry, reference, candidate)
    cv = np.asarray(cv)
    if cv.shape != (len(x), 12, 2) or not np.isfinite(cv).all():
        raise ValueError('Finite causal CV rollout required')
    x = np.column_stack((x, (cv/scale[:, None, None]).reshape(-1, 24), bits)).astype(np.float32)
    if x.shape != (len(cv), 383) or not np.isfinite(x).all():
        raise ValueError('Finite 383-dimensional bridge schema required')
    return x, rollout_envelope(reference, candidate)


def targets(cv_error, reference_error, candidate_error, easy_cut):
    utility, all_risk = relative_targets(cv_error, reference_error, candidate_error,
        reference='floor', event='all', easy_cut=easy_cut)
    _, easy_risk = relative_targets(cv_error, reference_error, candidate_error,
        reference='floor', event='easy', easy_cut=easy_cut)
    return dict(utility=utility, all_risk=all_risk, easy_risk=easy_risk)


def choices(utility, all_risk, easy_risk, moving, envelope, *, arm):
    moving, envelope = np.asarray(moving), np.asarray(envelope)
    n = len(moving)
    if (moving.dtype != bool or moving.shape != (n,) or envelope.shape != (n,)
            or not np.isfinite(envelope).all() or (envelope < 0).any()):
        raise ValueError('Aligned causal motion guard and envelope required')
    u, a, e = map(np.asarray, (utility, all_risk, easy_risk))
    if any(v.shape != (n, 2) or not np.isfinite(v).all() or (v < 0).any() for v in (u, a, e)):
        raise ValueError('Finite nonnegative aligned predicted costs required')
    if arm not in POLICIES:
        raise ValueError('Unregistered policy')
    if arm == 'reference_easy': return np.zeros(n, bool)
    if arm == 'candidate_all': return np.ones(n, bool)
    use = moving & (envelope > 0) & (u[:, 0] > u[:, 1])
    if arm in ('all_risk_only', 'dual_risk', 'ridge_dual'):
        use &= a[:, 1] <= .02*a[:, 0]
    if arm in ('easy_risk_only', 'dual_risk', 'ridge_dual'):
        use &= e[:, 1] <= .02*e[:, 0]
    return use


def scalar_replay(utility, all_risk, easy_risk, moving, envelope, arm):
    rows = []
    for u, a, e, move, env in zip(utility, all_risk, easy_risk, moving, envelope):
        if arm == 'reference_easy': use = False
        elif arm == 'candidate_all': use = True
        else:
            use = bool(move and env > 0 and u[0] > u[1])
            if arm in ('all_risk_only', 'dual_risk', 'ridge_dual'): use &= a[1] <= .02*a[0]
            if arm in ('easy_risk_only', 'dual_risk', 'ridge_dual'): use &= e[1] <= .02*e[0]
        rows.append(use)
    return np.asarray(rows, bool)
