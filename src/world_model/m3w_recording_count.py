"""Recording-aware supervision for causal counts over fixed recipient menus."""

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack
from torch import nn


def teacher(gain, risk, easy, legal, offset, easy_cap, mass, budget, seconds=2.):
    """Solve a training-recording label problem; no trajectory is emitted here."""
    q, c = np.nonzero(legal[:, 1:])
    c += 1
    counts = np.zeros(len(legal), dtype=np.int64)
    g, r, e = gain[q, c].T, risk[q, c].T, easy[q, c]
    if mass <= 0. or not e.any() or budget == 0:
        return counts, "no_support"
    n = len(q)
    col = np.arange(n)
    one = coo_array((np.ones(n), (q, col)), shape=(len(legal), n+1))
    scale = np.abs(r).max(1).clip(1.)
    matrix = vstack((one, csc_array(np.r_[c, 0.][None]),
        csc_array(np.column_stack((r/scale[:, None], np.zeros(3)))),
        csc_array(np.r_[e, 0.][None]),
        csc_array(np.column_stack((-g/mass, np.ones(2))))), format="csc")
    lo = np.r_[np.full(len(legal)+4, -np.inf), 1., -np.inf, -np.inf]
    hi = np.r_[np.ones(len(legal)), budget, 0., 0., easy_cap/scale[2], np.inf, np.array([0., offset])/mass]
    sol = milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(matrix, lo, hi), options=dict(time_limit=seconds, node_limit=400, mip_rel_gap=0.))
    if sol.x is not None:
        ix = sol.x[:-1] > .5
        net = r[:, ix].sum(1)
        if (len(np.unique(q[ix])) == ix.sum() and c[ix].sum() <= budget and e[ix].any()
                and net[0] <= 0. and net[1] <= 0. and net[2] <= easy_cap
                and 100.*(g[:, ix].sum(1)+[0., offset]).min()/mass > 1e-6):
            counts[q[ix]] = c[ix]
            return counts, "verified_witness"
    if sol.status == 2:
        return counts, "infeasible"
    dual = getattr(sol, "mip_dual_bound", None)
    if dual is not None and -100.*dual <= 1e-6:
        return counts, "nonpositive_bound"
    return counts, "unresolved"


def streaming_count(score, legal, recordings, starts, rate=.2, bank=2.):
    """A causal token cap; current history may change, the query order may not."""
    take = np.zeros(len(score), dtype=np.int64)
    last = None
    closed = set()
    tokens = 0.
    last_start = -np.inf
    for i, (s, legal_row, rec, t) in enumerate(zip(score, legal, recordings, starts, strict=True)):
        if rec != last:
            if rec in closed:
                raise ValueError("Recording must occupy a contiguous query sequence")
            closed.add(rec)
            last, last_start, tokens = rec, -np.inf, 0.
        if t <= last_start:
            raise ValueError("Expected strictly increasing observation starts")
        last_start = t
        if legal_row[1:].any():
            tokens = min(bank, tokens+rate)
        on = legal_row & (np.arange(3) <= tokens+1e-9)
        take[i] = np.where(on, s, -np.inf).argmax()
        tokens = max(0., tokens-take[i])
    return take


class CountHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 3))

    def forward(self, x, legal):
        return self.net(x).masked_fill(~legal, -1e4)


def objective(logits, target):
    weight = torch.where(target > 0, 4., 1.)
    return (nn.functional.cross_entropy(logits, target, reduction="none")*weight).sum()/weight.sum()
