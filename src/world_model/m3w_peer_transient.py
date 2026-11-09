"""Matched own-history and peer-history motion transients."""

import numpy as np

MIN_COS = .8
HALF_LIFE = 4.


def candidates(history, q, cv):
    """One additional path per row and arm, with identical slot eligibility."""
    if (history.shape != (len(q), 8, 2) or cv.shape != (len(q), 12, 2)
            or not np.isfinite(history).all() or not np.isfinite(cv).all()):
        raise ValueError('Require aligned, finite observed histories and CV paths')
    centered = history - history[:, -1:]
    time = np.arange(8.) - 3.5
    slow = np.einsum('t,ntd->nd', time, centered) / (time @ time)
    quick = -centered[:, -2]
    transient = quick - slow
    peers = np.full(len(q), -1, dtype=int)
    for j in np.unique(q):
        ix = np.flatnonzero(q == j)
        if len(ix) < 3:
            continue
        v = slow[ix]
        speed = np.linalg.norm(v, axis=-1)
        norm = speed[:, None] * speed[None, :]
        d = np.linalg.norm(centered[ix, 0] - centered[ix, -1], axis=-1)
        positions = history[ix, -1]
        distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        legal = ((norm > 0.) & ((v @ v.T) >= MIN_COS * norm)
                 & (distance <= (d[:, None] + d[None, :]) / 2.))
        np.fill_diagonal(legal, False)
        eligible = legal.any(axis=1)
        peers[ix[eligible]] = ix[np.where(legal, distance, np.inf).argmin(axis=1)[eligible]]

    steps = np.arange(1., 13.)
    decay = np.exp2(-steps / HALF_LIFE).cumsum()
    origin = slow[:, None] * steps[None, :, None]
    active = peers >= 0
    donor = np.where(active, peers, np.arange(len(q)))
    raw = np.stack((origin + transient[:, None] * decay[None, :, None],
                    origin + transient[donor, None] * decay[None, :, None]), axis=1)
    delta = raw - cv[:, None]
    peak = np.linalg.norm(delta, axis=-1).max(axis=-1)
    common = np.where(active, peak.min(axis=1), 0.)
    ratio = np.divide(common[:, None], peak, out=np.zeros_like(peak), where=peak > 0.)
    paths = cv[:, None] + delta * ratio[..., None, None]
    return paths, peers, common


def replace_recipients(cv, additive, extra, q):
    """On double edits, preserve original recipients and replace only eligible shapes."""
    peak = np.linalg.norm(additive - cv, axis=-1).max(axis=-1)
    cap = np.linalg.norm(extra - cv, axis=-1).max(axis=-1)
    _, group, counts = np.unique(q, return_inverse=True, return_counts=True)
    edits = np.bincount(group, weights=(peak > 0.), minlength=len(counts))
    use = (edits[group] == 2) & (peak > 0.) & (cap > 0.) & (cap >= peak - 1e-10)
    result = additive.copy()
    result[use] = cv[use] + (extra[use] - cv[use]) * np.minimum(1., peak[use] / cap[use])[:, None, None]
    np.testing.assert_allclose(np.linalg.norm(result - cv, axis=-1).max(axis=-1), peak, rtol=1e-9, atol=1e-9)
    return result, use
