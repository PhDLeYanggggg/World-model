"""Fixed causal prefix-risk policies. No outcome availability enters decisions."""
import numpy as np
from src.evaluation.m3w_conditional_cost_audit import strict_bits

POLICIES = ('control_terminal', 'profile_terminal', 'profile_guard', 'control_matched', 'profile_matched')


def selections(control, profile, past, distance, ids):
    n = len(ids)
    if (control.shape != (n, 12, 2) or profile.shape != control.shape or distance.shape != (n, 12)
            or not np.isfinite(control).all() or not np.isfinite(profile).all()
            or (control < 0).any() or (profile < 0).any()
            or len(np.unique(ids)) != n or not np.isfinite(distance).all() or (distance < 0).any()):
        raise ValueError('Finite causal profiles, aligned disagreement and unique ids required')
    ct = strict_bits(control[:, -1], past, distance[:, -1])
    pt = strict_bits(profile[:, -1], past, distance[:, -1])
    guard = pt & np.all(profile[..., 1] <= .1*profile[..., 0], axis=1)
    support = (distance[:, -1] > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
    def matched(score):
        pool = np.flatnonzero(support)
        gain = score[:, -1, 0] - score[:, -1, 1]
        out = np.zeros(n, bool)
        out[pool[np.lexsort((ids[pool], -gain[pool]))[:int(guard.sum())]]] = True
        return out
    return dict(control_terminal=ct, profile_terminal=pt, profile_guard=guard,
                control_matched=matched(control), profile_matched=matched(profile))
