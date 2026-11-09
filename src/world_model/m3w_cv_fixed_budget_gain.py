"""Missing-step-aware utility and count-preserving query optimization."""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_cv_risk_frontier import round_support


def missing_terms(cv, ols, path, scale):
    """Per-step gain when the outcome is missing, relative to keeping CV."""
    return np.stack((-np.linalg.norm(path-cv, axis=-1),
        np.linalg.norm(ols-cv, axis=-1)-np.linalg.norm(ols-path, axis=-1)), -1) / scale[:, None, None]


def targets(cv, ols, path, y, valid, scale):
    safe = np.where(valid[..., None], y, 0.)
    benefit = np.linalg.norm(cv-safe, axis=-1)-np.linalg.norm(path-safe, axis=-1)
    observed = np.where(valid, benefit, 0.).mean(1) / scale
    geom = missing_terms(cv, ols, path, scale)
    missing = ~valid
    gain = observed[:, None] + (missing[..., None]*geom).mean(1)
    return tuple(x.astype(np.float32) for x in (geom, missing, observed, gain))


class Head(nn.Module):
    def __init__(self, past_width, width):
        super().__init__()
        self.past_width = past_width
        self.mask = nn.Sequential(nn.Linear(past_width, 64), nn.SiLU(), nn.Linear(64, 12))
        self.path = nn.Sequential(nn.Linear(width, 96), nn.SiLU(), nn.Linear(96, 64),
                                  nn.SiLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.mask(x[:, :self.past_width]), self.path(x)


def decode(mask, path, geom, factorized):
    if factorized:
        return path[:, 2:] + (mask.sigmoid()[..., None]*geom).mean(1)
    return path[:, :2]


def objective(net, x, geom, missing, observed, gain, factorized):
    mask, path = net(x)
    pred = decode(mask, path, geom, factorized)
    return (F.mse_loss(pred, gain) + .25*F.mse_loss(path[:, 2], observed)
            + .1*F.binary_cross_entropy_with_logits(mask, missing))


class Budget:
    """One legal action per agent, exactly the incumbent intervention count."""

    def __init__(self, legal, risk, reference, cv_gain, anchor):
        n = len(anchor)
        self.n = n
        self.count = int((anchor >= 0).sum())
        self.row, action = np.nonzero(np.column_stack((np.ones(n, bool), legal)))
        self.action = action-1
        self.edit = action > 0
        self.offsets = np.r_[0, np.bincount(self.row, minlength=n).cumsum()]
        self.anchor = np.flatnonzero(self.action == anchor[self.row])
        if len(self.anchor) != n:
            raise ValueError('Illegal incumbent action')
        row, arm = self.row[self.edit], self.action[self.edit]
        col = np.zeros((len(self.row), 6))
        col[self.edit] = np.column_stack((risk[row, arm], -reference[row], -cv_gain[row, arm]))
        limits = col[self.anchor].sum(0)
        units = np.maximum(np.maximum(np.abs(limits), np.max(np.abs(col), axis=0)), 1.)
        self.col = col/units
        self.limit = self.col[self.anchor].sum(0)
        self.equal = sparse.coo_matrix(
            (np.ones(len(self.row)+len(row)), (np.r_[self.row, np.full(len(row), n)],
             np.r_[np.arange(len(self.row)), np.flatnonzero(self.edit)])),
            shape=(n+1, len(self.row))).tocsr()

    def solve(self, utility):
        if not self.count:
            return self.action[self.anchor]
        score = np.zeros(len(self.row))
        score[self.edit] = np.asarray(utility)[self.row[self.edit], self.action[self.edit]]
        score /= max(1., np.abs(score).max())
        obj = linprog(-score, A_ub=self.col.T, b_ub=self.limit,
            A_eq=self.equal, b_eq=np.r_[np.ones(self.n), self.count], bounds=(0., 1.), method='highs')
        if not obj.success:
            raise RuntimeError(f'Incumbent-feasible LP: {obj.message}')
        quality = np.column_stack((score, score, self.col, self.edit, -self.edit.astype(float)))
        pool = round_support(obj.x, self.offsets, quality, np.r_[self.limit, self.count, -self.count])
        best = self.anchor
        for p in pool:
            if (self.edit[p].sum() == self.count and np.all(self.col[p].sum(0) <= self.limit+1e-7)
                    and score[p].sum() > score[best].sum()+1e-10):
                best = p
        return self.action[best]
