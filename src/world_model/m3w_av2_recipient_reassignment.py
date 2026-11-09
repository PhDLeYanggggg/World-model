"""Assign two frozen edit profiles to different actors without changing their radii."""

import numpy as np

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.world_model import m3w_av2_history_replay as history


ARMS = ("fixed_actors", "history_gain", "history_support", "history_joint")
PAIR_WEIGHT = .5


def assignments(n):
    return np.argwhere(~np.eye(n, dtype=bool))


def bank(cv, h, anchor):
    slots = np.flatnonzero(np.any(anchor != cv, axis=(1, 2)))
    if len(slots) != 2:
        raise ValueError("Require exactly two nonzero edit profiles")
    d = anchor[slots]-cv[slots]
    axis = h[:, -1]-h[:, 0]
    x, y = axis[slots][None], axis[:, None]
    valid = (np.linalg.norm(x, axis=-1) > 1e-8) & (np.linalg.norm(y, axis=-1) > 1e-8)
    angle = np.where(valid, np.arctan2(x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0], (x*y).sum(-1)), 0.)
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    p = cv[:, None]+np.stack((c*d[..., 0]-s*d[..., 1], s*d[..., 0]+c*d[..., 1]), axis=-1)
    p[slots, np.arange(2)] = anchor[slots]
    expected = np.linalg.norm(d, axis=-1)
    np.testing.assert_allclose(np.linalg.norm(p-cv[:, None], axis=-1), np.broadcast_to(expected, p.shape[:-1]), atol=1e-8, rtol=1e-8)
    if not np.any(p != cv[:, None], axis=(2, 3)).all():
        raise ValueError("Nonzero slot collapsed after reassignment")
    return p, slots


def render(cv, p, recipients):
    out = cv.copy()
    out[recipients] = p[recipients, np.arange(2)]
    return out


def select(h, cv, p, slots, opts, easy_cut):
    i, j = opts.T
    delta = p-cv[:, None]
    velocity = h[:, -1]-h[:, -2]
    gain, pair = np.zeros(delta.shape[:2]), np.zeros(len(opts))
    votes = np.zeros(len(h), dtype=int)
    for k in history.PREFIXES:
        cv0, ols0, _, scale = history.motion.rollout(h[:, :k], 8-k)
        past = h[:, k:]-h[:, k-1:k]
        v = h[:, k-1]-h[:, k-2]
        change = history.align(delta[:, :, :8-k], velocity[:, None], v[:, None])
        c = cv0-past
        e = c[:, None]+change
        a, b, o = (np.linalg.norm(t, axis=-1).mean(-1) for t in (c, e, ols0-past))
        gain += np.minimum(a[:, None]-b, o[:, None]-b)/scale[:, None]
        base_pair = np.linalg.norm(c[i]-c[j], axis=-1).mean(1)
        revised_pair = np.linalg.norm(e[i, 0]-e[j, 1], axis=-1).mean(1)
        pair += (base_pair-revised_pair)/((scale[i]+scale[j])*.5)
        votes += (a > 0.) & (a <= easy_cut*(9-k)/13.)
    proxy_easy = votes >= (len(history.PREFIXES)+1)//2
    available = proxy_easy[i] | proxy_easy[j]
    same = (np.sort(opts, axis=1) == slots).all(1)
    additive = gain[i, 0]+gain[j, 1]
    return np.array([np.where(same, additive, -np.inf).argmax(), additive.argmax(),
        np.where(available if available.any() else True, additive, -np.inf).argmax(),
        np.where(available if available.any() else True, additive+PAIR_WEIGHT*pair, -np.inf).argmax()]), bool(available.any())


def eligibility(cv, ols, p, y, valid, easy_cut, opts):
    """Exact standalone feasibility of each ordered two-slot assignment."""
    mass, fallback, gain, _ = candidate_terms(cv, ols, p, y, valid)
    risk = risk_excess(cv, p, y, valid, easy_cut)
    safe = np.where(valid[..., None], y, 0.)
    c = np.where(valid, np.linalg.norm(cv-safe, axis=-1), 0.).sum(1)/valid.sum(1).clip(1)
    e = np.where(valid[:, None], np.linalg.norm(p-safe[:, None], axis=-1), 0.).sum(2)/valid.sum(1).clip(1)[:, None]
    easy = (c > 0.) & (c <= easy_cut)
    signed = np.where(easy[:, None], e-c[:, None], 0.)
    i, j = opts.T
    total_gain = gain[i, 0]+gain[j, 1]+np.array([0., fallback.sum()])
    supported = (easy[i] | easy[j]) & (mass.sum() > 0.)
    return supported & (total_gain > 0.).all(1) & ((risk[i, 0]+risk[j, 1]) <= 0.).all(1) & (
        signed[i, 0]+signed[j, 1] <= .02*c[easy].sum()), supported
