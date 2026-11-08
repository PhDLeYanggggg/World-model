"""Past-only bridge gains and separate safety/support events."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_signed_bridge_risk as signed

SAFETY = (.35, .50, .65, .80)
EASY = (.05, .15, .30)


def targets(m):
    good = (signed.signed_target(m) <= 0.).all(2) & (m[..., 4] > 0.)
    return np.concatenate((m[..., :2], good[..., None], (good & (m[..., 5] > 0.))[..., None]), axis=2).astype(np.float32)


class Hurdle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(signed.base.WIDTH, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.net(x)


def objective(out, truth, legal):
    p, t = out[legal], truth[legal]
    value = nn.functional.smooth_l1_loss(p[:, :2], t[:, :2], beta=.04)
    event = nn.functional.binary_cross_entropy_with_logits(p[:, 2:], t[:, 2:])
    return value + .03*event


def bank_decode(pred, legal, queries, policies):
    """Maximize conservative gain proxy over singleton and pair actions."""
    if not np.isfinite(pred).all():
        raise ValueError("Predictions must be finite")
    out = np.full((len(policies), len(queries)), -1, dtype=int)
    thresholds = np.asarray(policies, dtype=float)
    for q in np.unique(queries):
        ix = np.flatnonzero(queries == q)
        if len(ix) < 2:
            continue
        j, a = np.nonzero(legal[ix])
        if not len(j):
            continue
        m = pred[ix[j], a]
        left = np.arange(len(m))
        right = np.full(len(m), -1, dtype=int)
        if len(ix) > 2:
            k, l = np.triu_indices(len(m), 1)
            different = j[k] != j[l]
            left = np.r_[left, k[different]]
            right = np.r_[right, l[different]]
        pair = right >= 0
        score = (m[left, :2]+np.where(pair[:, None], m[right, :2], 0.)).min(1)
        safety = np.minimum(m[left, 2], np.where(pair, m[right, 2], 1.))
        easy = np.maximum(m[left, 3], np.where(pair, m[right, 3], 0.))
        legal_policy = (safety >= thresholds[:, :1]) & (easy >= thresholds[:, 1:2])
        ranking = np.where(legal_policy, score, -np.inf)
        best = ranking.argmax(1)
        for row in np.flatnonzero(ranking[np.arange(len(best)), best] > 0.):
            i, k = left[best[row]], right[best[row]]
            out[row, ix[j[i]]] = a[i]
            if k >= 0:
                out[row, ix[j[k]]] = a[k]
    return out


def matched_independent(pred, legal, queries, chosen):
    a = np.where(legal, pred[..., :2].min(2), -np.inf)
    result = np.full(len(queries), -1, dtype=int)
    for q in np.unique(queries):
        ids = np.flatnonzero(queries == q)
        count = int((chosen[ids] >= 0).sum())
        if count:
            arm = a[ids].argmax(1)
            v = a[ids, arm]
            at = np.argsort(-v, kind="stable")[:count]
            if not np.isfinite(v[at]).all():
                raise ValueError("Cannot reproduce the selected count")
            result[ids[at]] = arm[at]
    return result
