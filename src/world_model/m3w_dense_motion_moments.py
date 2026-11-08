"""Learn CV-relative costs for broad, bounded OLS4 interpolation."""

import numpy as np
import torch
from torch import nn


LEVELS = np.array([0., .25, .5, .75, 1.])
MULTIPLIERS = (0., .25, .5, 1., 2., 4., 8., 16., 32., 64., 128., 256.)


def slope(h):
    t = np.arange(h.shape[1], dtype=float)
    t -= t.mean()
    return np.einsum("t,ntd->nd", t, h-h[:, -1:])/(t@t)


def features(history):
    h = np.asarray(history, float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError("Eight finite causal positions required")
    d = np.diff(h, axis=1)
    scale = np.sqrt(np.square(d).sum(2).mean(1)).clip(.05)
    v = slope(h[:, -4:])
    axis = v/np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
    local = np.stack((np.einsum("nd,ntd->nt", axis, d),
                      np.einsum("nd,ntd->nt", axis[:, ::-1]*[-1., 1.], d)), axis=-1)
    hs = []
    for end in (4, 5, 6, 7):
        q = h[:, end-4:end]
        lead = np.arange(1, 9-end)[None, :, None]
        target = h[:, end:]-q[:, -1:]
        before = np.linalg.norm((q[:, -1:]-q[:, -2:-1])*lead-target, axis=-1).mean(1)
        after = np.linalg.norm(slope(q)[:, None]*lead-target, axis=-1).mean(1)
        hs.append((before-after)/scale)
    hind = np.stack(hs, axis=1)
    bound = np.linalg.norm(v-d[:, -1], axis=1)/scale
    x = np.column_stack(((local/scale[:, None, None]).reshape(len(h), 14),
                         np.log1p(scale), np.linalg.norm(d, axis=2).std(1)/scale,
                         np.linalg.norm(v, axis=1)/scale, bound, hind))
    return x.clip(-16, 16).astype(np.float32), 12.*scale, bound, hind.mean(1)


def make_path(cv, ols, actions):
    a = LEVELS[np.asarray(actions)]
    path = cv+a[:, None, None]*(ols-cv)
    path[a == 1.] = ols[a == 1.]
    return path


def targets(cv, ols, y, valid, normalizer, easy_cut):
    """Compute reference, available-label gain and conservative harm moments."""
    from src.evaluation.m3w_selection_forecast_readout import errors

    known = valid.any(1)
    base, _ = errors(cv, y, valid)
    easy = known & (base > 0) & (base <= easy_cut)
    ref = np.stack((np.where(known, base, 0.), np.where(easy, base, 0.)), axis=1)
    delta = np.linalg.norm(ols-cv, axis=-1).max(1)
    moments = []
    for j, a in enumerate(LEVELS[1:], 1):
        p = make_path(cv, ols, np.full(len(cv), j))
        error, _ = errors(p, y, valid)
        gain = np.where(known, base-error, -a*delta)
        harm = np.maximum(-gain, 0.)
        moments.append(np.stack((gain, harm, np.where(easy | ~known, harm, 0.)), axis=1))
    return (ref/normalizer[:, None]).astype(np.float32), (
        np.stack(moments, axis=1)/normalizer[:, None, None]).astype(np.float32)


class DenseMoments(nn.Module):
    def __init__(self, width=22):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 48),
                                     nn.SiLU(), nn.Linear(48, 14))

    def forward(self, x, bound):
        z = self.network(x)
        all_mass = torch.nn.functional.softplus(z[:, 0])
        ref = torch.stack((all_mass, all_mass*z[:, 1].sigmoid()), dim=1)
        q = z[:, 2:].reshape(len(x), 4, 3)
        limits = bound[:, None]*torch.arange(1, 5, device=x.device, dtype=x.dtype)[None]/4
        gain = q[..., 0].tanh()*limits
        harm = q[..., 1].sigmoid()*limits
        easy = q[..., 2].sigmoid()*harm
        return ref, torch.stack((gain, harm, easy), dim=2)


def choose(ref, values, offsets, legal, budget=.02):
    """Select a feasible member of the fixed predicted-risk price ladder."""
    n = len(ref)
    gain, harm, easy_harm = values.transpose(2, 0, 1)
    all_excess = harm-budget*ref[:, :1]
    easy_excess = easy_harm-budget*ref[:, 1:]
    w = np.repeat(1./np.diff(offsets), np.diff(offsets))
    w /= w.mean()
    weighted = w[:, None]*gain
    best, best_mass, chosen_price = np.zeros(n, dtype=np.uint8), 0., None
    idx = np.arange(n)
    independent = None
    for price in MULTIPLIERS:
        score = np.where(legal[:, None] & (gain > 0), weighted-price*(all_excess+easy_excess), -np.inf)
        j = score.argmax(1)
        on = score[idx, j] > 0.
        action = np.where(on, j+1, 0).astype(np.uint8)
        if price == 0.:
            independent = action
        r = ref[on].sum(0)
        if np.any(r <= 0.):
            continue
        harms = np.array([harm[on, j[on]].sum(), easy_harm[on, j[on]].sum()])
        total = float(weighted[on, j[on]].sum())
        if np.all(harms <= budget*r) and total > best_mass:
            best, best_mass, chosen_price = action, total, price
    return independent, best, chosen_price


def matched_hindcast(actions, hind, offsets, legal):
    """Same query-wise amplitudes, allocated by a fixed past-only backtest."""
    result = np.zeros_like(actions)
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        values = np.sort(actions[lo:hi][actions[lo:hi] > 0])[::-1]
        rows = lo+np.flatnonzero(legal[lo:hi])
        top = rows[np.argsort(-hind[rows], kind="stable")[:len(values)]]
        if len(top) != len(values):
            raise ValueError("Matched control cannot preserve action support")
        result[top] = values
    return result
