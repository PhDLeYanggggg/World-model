"""Structured teacher for fixed-slot recipient and shape decisions."""
import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack
from torch import nn


def teacher(rows, slots, gain, offset, risk, easy, limit, constrained):
    """Maximize the worse of the two baseline-relative gains at fixed slots."""
    n = len(rows)
    if not n:
        return None, 'empty'
    if constrained and not easy.any():
        return None, 'no_easy'
    _, r = np.unique(rows, return_inverse=True)
    _, s = np.unique(slots, return_inverse=True)
    nr, ns = int(r.max()+1), int(s.max()+1)
    group = lambda ids, size: coo_array((np.ones(n), (ids, np.arange(n))), shape=(size, n+1))
    scale = max(np.abs(gain).max(), np.abs(offset).max(), 1e-8)
    pieces = [group(r, nr), group(s, ns), csc_array(np.c_[-gain.T/scale, np.ones(2)])]
    lower = np.r_[np.full(nr, -np.inf), np.ones(ns), -np.inf, -np.inf]
    upper = np.r_[np.ones(nr+ns), offset/scale]
    if constrained:
        weight = np.maximum(np.abs(risk).max(0), np.abs(limit)).clip(1e-8)
        pieces += [csc_array(np.c_[risk.T/weight[:, None], np.zeros(3)]), csc_array(np.r_[easy, 0.][None])]
        lower = np.r_[lower, [-np.inf]*3, 1.]
        upper = np.r_[upper, limit/weight, np.inf]
    result = milp(np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(vstack(pieces, format='csc'), lower, upper),
        options=dict(time_limit=3., mip_rel_gap=0.))
    if result.status == 2:
        return None, 'infeasible'
    if result.status != 0:
        return None, 'unresolved'
    take = np.flatnonzero(result.x[:n] > .5)
    if len(take) != ns or len(np.unique(r[take])) != ns or len(np.unique(s[take])) != ns:
        raise ValueError('Teacher violated slot assignment constraints')
    if constrained and (not easy[take].any() or (risk[take].sum(0) > limit+1e-8).any()):
        return None, 'unresolved'
    if np.min(gain[take].sum(0)+offset) <= 1e-10:
        return None, 'nonpositive'
    return take, 'qualified'


class Ranker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def objective(net, chosen, other):
    return nn.functional.softplus(net(other)-net(chosen)).mean()
