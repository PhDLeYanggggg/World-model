"""Query-budget upper envelopes for paired, missing-step-conservative utility."""

import numpy as np


def candidate_terms(cv, ols, paths, target, valid):
    """Paired lower numerators for fallback and for candidate actions."""
    safe = np.where(valid[..., None], target, 0.)
    c = np.linalg.norm(cv - safe, axis=-1)
    o = np.linalg.norm(ols - safe, axis=-1)
    p = np.linalg.norm(paths - safe[:, None], axis=-1)
    oc = np.linalg.norm(ols - cv, axis=-1)
    op = np.linalg.norm(ols[:, None] - paths, axis=-1)
    cp = np.linalg.norm(cv[:, None] - paths, axis=-1)
    cv_action = np.where(valid[:, None], c[:, None] - p, -cp).sum(-1)
    ols_action = np.where(valid[:, None], o[:, None] - p, -op).sum(-1)
    ols_fallback = np.where(valid, o - c, -oc).sum(-1)
    mass = np.where(valid, c, 0.).sum(-1)
    arbitrary = np.stack((mass, mass + np.where(valid, 0., oc).sum(-1)), axis=1)
    gain = np.stack((cv_action, ols_action - ols_fallback[:, None]), axis=-1)
    return mass, ols_fallback, gain, arbitrary


def risk_excess(cv, paths, target, valid, easy_cut):
    """Linear excesses for the original all and easy positive-harm tests."""
    count = valid.sum(1)
    known = count > 0
    safe = np.where(valid[..., None], target, 0.)
    c = np.where(valid, np.linalg.norm(cv - safe, axis=-1), 0.).sum(1) / count.clip(1)
    p = np.where(valid[:, None], np.linalg.norm(paths - safe[:, None], axis=-1), 0.).sum(-1) / count.clip(1)[:, None]
    radius = np.linalg.norm(paths - cv[:, None], axis=-1).max(-1)
    selected = radius > 0
    easy = known & (c > 0) & (c <= easy_cut)
    change = np.maximum(p - c[:, None], 0.)
    unknown = np.where(known[:, None], 0., radius)
    ref = c[:, None] * selected
    return np.stack((change + unknown - .02 * ref,
                     np.where(easy[:, None], change - .02 * ref, 0.) + unknown), axis=-1)


def row_ranks(query):
    order = np.argsort(query, kind='stable')
    counts = np.unique(query, return_counts=True)[1]
    return order, np.r_[0, counts.cumsum()]


def cap_curve(score, order, offsets):
    """Exact per-query top-k sum with the no-edit option."""
    by_rank = np.zeros(int(np.diff(offsets).max()))
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        v = np.sort(np.maximum(score[order[a:b]], 0.))[::-1]
        by_rank[:len(v)] += v
    return np.r_[0., np.cumsum(by_rank)]


def minimum_cap(curve, intercept):
    where = np.flatnonzero(curve + intercept > 0.)
    return int(where[0]) if len(where) else None


def choice_at_cap(score, order, offsets, cap):
    winner = score.argmax(1)
    picked = score[np.arange(len(winner)), winner]
    choice = np.full(len(winner), -1)
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        local = order[a:b]
        local = local[np.argsort(-picked[local], kind='stable')[:cap]]
        use = local[picked[local] > 0.]
        choice[use] = winner[use]
    return choice
