"""Learn when a fixed-budget recipient replacement has useful options."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from torch import nn

LOWER_SHARE = .95


def describe(x):
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite matrix of causal swap descriptors')
    if not len(x):
        return np.zeros(3*x.shape[1]+1, dtype=np.float32)
    return np.r_[x.mean(0), x.std(0), x.max(0), np.log1p(len(x))].astype(np.float32)


def network(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 48), nn.SiLU(),
                         nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1), nn.Flatten(0))


def allocate(score, cost, incumbent):
    """Offline same-count allocation within 95--100% of the incumbent's cost."""
    if (score.ndim != 1 or cost.shape != score.shape or incumbent.shape != score.shape
            or incumbent.dtype != np.bool_ or not np.isfinite(score).all()
            or not np.isfinite(cost).all() or (cost < 0.).any() or (cost[incumbent] <= 0.).any()):
        raise ValueError('Invalid candidate allocation')
    k, budget = int(incumbent.sum()), float(cost[incumbent].sum())
    if k == 0:
        return incumbent.copy(), 'zero_quota'
    legal = (cost > 0.) & (cost <= budget)
    if k == 1:
        j = int(np.argmax(np.where(legal & (cost >= LOWER_SHARE*budget), score, -np.inf)))
        out = np.arange(len(score)) == j
    else:
        weights = cost/budget
        constraint = LinearConstraint(np.stack((np.ones_like(weights), weights)), [k, LOWER_SHARE], [k, 1.])
        values = score - score[legal].min()
        values /= max(1., abs(values[legal]).max())
        res = milp(-values, integrality=np.ones(len(score)), bounds=Bounds(0., legal.astype(float)),
                   constraints=constraint, options=dict(time_limit=3., mip_rel_gap=0.))
        if res.status != 0 or res.x is None:
            return incumbent.copy(), 'solver_fallback'
        out = res.x > .5
    spend = float(cost[out].sum())/budget
    if (int(out.sum()) != k or not legal[out].all() or not LOWER_SHARE-1e-7 <= spend <= 1.+1e-7
            or score[out].sum()+1e-7 < score[incumbent].sum()):
        return incumbent.copy(), 'postcheck_fallback'
    return out, 'optimal'
