"""Frozen predictor-family transfer choices; no outcome inputs or fitted gates."""
import numpy as np
from src.world_model.m3w_bounded_cost_head import ARMS


def choices(scores, past, distance, ids):
    past, distance, ids = map(np.asarray, (past, distance, ids))
    n = len(ids)
    if (set(scores) != set(ARMS) or past.shape != (n, 8, 2)
            or distance.shape != (n,) or len(np.unique(ids)) != n
            or not np.isfinite(past).all() or not np.isfinite(distance).all()
            or (distance < 0).any()):
        raise ValueError('Aligned finite past context and unique query IDs required')
    allowed = (distance > 0) & np.any(past[:, -1] != past[:, -2], axis=1)
    result = dict(floor=np.zeros(n, bool), uncontrolled=np.ones(n, bool), past_stop=allowed.copy())
    for arm in ARMS:
        s = np.asarray(scores[arm])
        if s.shape != (n, 2) or not np.isfinite(s).all() or (s < 0).any():
            raise ValueError('Finite nonnegative continuous benefit/harm scores required')
        result[arm+'_net_stop'] = allowed & (s[:, 0] > s[:, 1])
        result[arm+'_strict_stop'] = result[arm+'_net_stop'] & (s[:, 1] <= .1*s[:, 0])
    count = int(result['bounded_fraction_strict_stop'].sum())
    pool = np.flatnonzero(allowed)
    for arm in ARMS:
        net = scores[arm][:, 0]-scores[arm][:, 1]
        order = pool[np.lexsort((ids[pool], -net[pool]))[:count]]
        result[arm+'_matched_count'] = np.zeros(n, bool)
        result[arm+'_matched_count'][order] = True
    return result
