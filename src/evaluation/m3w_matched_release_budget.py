"""Match selection count and two causal movement budgets for fixed proposals."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


TOLERANCE = .01


def movement(cv, rows, paths, weight):
    if (rows.ndim != 2 or rows.shape[1] != 2 or rows.dtype.kind not in 'iu'
            or len(np.unique(rows)) != rows.size or np.any(rows < 0) or np.any(rows >= len(cv))
            or paths.shape != (*rows.shape, 12, 2) or weight.shape != (len(cv),)
            or not np.isfinite(weight).all() or np.any(weight <= 0.)):
        raise ValueError('Expected disjoint two-actor proposals and observed query weights')
    lengths = np.linalg.norm(paths-cv[rows], axis=-1).sum(axis=-1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.):
        raise ValueError('Both actors must move away from the fallback')
    return np.stack((lengths.sum(axis=1), (lengths*weight[rows]).sum(axis=1)))


def select(scores, costs, original, seconds=30.):
    if (scores.ndim != 1 or costs.shape != (2, len(scores)) or original.shape != scores.shape
            or original.dtype != bool or not original.any() or not np.isfinite(scores).all()
            or not np.isfinite(costs).all() or np.any(costs <= 0.)):
        raise ValueError('Expected finite causal scores, positive costs and an original quota')
    targets = costs[:, original].sum(axis=1)
    scaled = costs/targets[:, None]
    k = int(original.sum())
    utility = scores-scores.mean()
    scale = max(float(abs(utility).max()), 1e-12)
    sol = milp(-utility/scale, integrality=np.ones(len(scores)), bounds=Bounds(0., 1.),
               constraints=LinearConstraint(np.vstack((np.ones(len(scores)), scaled)),
                                            [k, 1.-TOLERANCE, 1.-TOLERANCE],
                                            [k, 1.+TOLERANCE, 1.+TOLERANCE]),
               options=dict(time_limit=seconds, mip_rel_gap=1e-9))
    if not sol.success or sol.x is None:
        raise RuntimeError(f'Movement-matching solve is unresolved: status {sol.status}')
    chosen = sol.x > .5
    if chosen.sum() != k or np.any(abs(scaled[:, chosen].sum(axis=1)-1.) > TOLERANCE+1e-8):
        raise ValueError('Solver failed the exact count or causal spend check')
    if float(utility[chosen].sum())+1e-7*scale < float(utility[original].sum()):
        raise ValueError('Solver objective is below the feasible original mask')
    return chosen
