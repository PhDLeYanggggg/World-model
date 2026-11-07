"""Label-aware feasibility bounds for a fixed recording and affine slate."""
import numpy as np

from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments
from src.evaluation.m3w_selection_forecast_readout import policy


def min_affine_cost(residual, delta, valid):
    """Minimize masked mean norm, or max norm for a wholly unknown row.

    Bracket the minimizer using a valid subgradient. The Lipschitz bound
    also applies at zero residuals and tied maximal steps.
    """
    known = valid.any(1)
    count = np.maximum(valid.sum(1), 1)
    speed = np.linalg.norm(delta, axis=-1)
    lipschitz = np.where(known, (valid*speed).sum(1)/count, speed.max(1))

    def value(alpha):
        p = residual+alpha[:, None, None]*delta
        norm = np.linalg.norm(p, axis=-1)
        slope = np.divide((p*delta).sum(2), norm, out=np.zeros_like(norm), where=norm > 0.)
        peak = norm.argmax(1)
        return (np.where(known, (valid*norm).sum(1)/count, norm.max(1)),
                np.where(known, (valid*slope).sum(1)/count, slope[np.arange(len(norm)), peak]))

    lo, hi = np.zeros(len(valid)), np.ones(len(valid))
    for _ in range(42):
        mid = (lo+hi)/2.
        _, g = value(mid)
        lo, hi = np.where(g < 0., mid, lo), np.where(g < 0., hi, mid)
    # Include endpoints: boundary optima and exact identity must remain visible.
    grid = np.stack((lo, hi, np.zeros_like(lo), np.ones_like(hi)))
    costs = np.stack([value(a)[0] for a in grid])
    ix = np.arange(len(lo))
    best = costs.argmin(0)
    upper = costs[best, ix]
    lower = np.maximum(0., value(lo)[0]-lipschitz*(hi-lo))
    return grid[best, ix], np.minimum(lower, upper), upper


def audit(cv, reference, proposal, target, valid, switched, recording, easy_cut, hard_cut):
    """Return a checked witness or a conservative exclusion, not a policy."""
    reference, proposal = np.asarray(reference, float), np.asarray(proposal, float)
    mask = np.asarray(switched, bool)
    if any(not np.array_equal((x != cv).any((1, 2)), mask) for x in (reference, proposal)):
        raise ValueError('Both endpoints must preserve the original recipients')
    m = moments(cv, reference, target, valid, easy_cut)
    at, v = m['known'][mask], valid[mask]
    base, easy = m['base'][mask], m['easy'][mask]
    all_mass, easy_mass, full_easy = base.sum(), base[easy].sum(), m['ref_easy'].sum()
    support = np.array([all_mass, easy_mass, full_easy])
    cuts = BUDGET*support
    if (support <= 0.).any():
        return dict(status='structural_support_absent', switches=int(mask.sum()),
                    support_px=support.tolist()), None
    r = reference[mask]
    d = proposal[mask]-r
    truth = np.where(v[..., None], target[mask], 0.)
    residual = r-np.where(at[:, None, None], truth, cv[mask])
    alpha, lower, upper = min_affine_cost(residual, d, v)

    def channels(cost):
        unknown = float(cost[~at].sum())
        harm = np.maximum(cost[at]-base[at], 0.).sum()
        easy_harm = np.maximum(cost[easy]-base[easy], 0.).sum()
        return np.array([harm+unknown, easy_harm+unknown, (cost[easy]-base[easy]).sum()])

    min_lower, min_upper = channels(lower), channels(upper)
    tol = 1e-9*(1.+np.abs(cuts))
    result = dict(status='unresolved', switches=int(mask.sum()), support_px=support.tolist(),
                  minimum_excess_lower_px=(min_lower-cuts).tolist(),
                  minimum_excess_upper_px=(min_upper-cuts).tolist(),
                  unknown_radius_lower_px=float(lower[~at].sum()))
    if np.any(min_lower > cuts+tol):
        result.update(status='risk_infeasible_with_fixed_recipients',
                      excluding_channels=(min_lower > cuts+tol).tolist(),
                      unknown_radius_alone_excludes_easy=bool(lower[~at].sum() > cuts[1]+tol[1]))
        return result, None

    # If the relaxed minimizer is identity, move infinitesimally toward the
    # original path and require exact nonidentity plus the original scorer.
    recs = np.full(len(mask), recording)
    for nudge in (0., 1e-12, 1e-10, 1e-8):
        p = reference.copy()
        p[mask] = r+((1.-nudge)*alpha)[:, None, None]*d
        if not np.array_equal((p != cv).any((1, 2)), mask):
            continue
        risk = policy(p, cv, cv, target, valid, recs, easy_cut=easy_cut,
                      hard_cut=hard_cut)['recordings'][recording]
        if risk['supported']:
            return dict(result, status='checked_feasible', risk=risk, identity_nudge=nudge), p
    return result, None


def safe_segment(anchor, candidate, cv, target, valid, switched, recording, easy_cut, hard_cut):
    """Find a safe point on one prechosen ray; no global pair optimality claim."""
    recs = np.full(len(valid), recording)
    delta = candidate-anchor

    def check(t):
        path = anchor+t*delta
        if not np.array_equal((path != cv).any((1, 2)), switched):
            return None
        risk = policy(path, cv, cv, target, valid, recs, easy_cut=easy_cut,
                      hard_cut=hard_cut)['recordings'][recording]
        return path if risk['supported'] else None

    out = check(1.)
    if out is not None:
        return 1., out
    lo, hi, out = 0., 1., anchor
    for _ in range(24):
        mid = (lo+hi)/2.
        p = check(mid)
        if p is None:
            hi = mid
        else:
            lo, out = mid, p
    return lo, out
