"""Prefix-validated one-step motion coupling, with a matched self control."""

import numpy as np

from src.world_model.m3w_peer_transient import HALF_LIFE, MIN_COS

FIT_PAIRS = 3
RIDGE = .25
IMPULSE_HALF_LIFE = 2.


def peers_in_prefix(history, query):
    """Choose the neighbor before the two held observed steps."""
    h = history[:, :6]
    progress = h[:, -1] - h[:, 0]
    distance_travelled = np.linalg.norm(progress, axis=-1)
    peers = np.full(len(h), -1, dtype=int)
    for k in np.unique(query):
        ix = np.flatnonzero(query == k)
        if len(ix) < 3:
            continue
        travel = distance_travelled[ix]
        scale = travel[:, None] * travel[None, :]
        dist = np.linalg.norm(h[ix, None, -1] - h[None, ix, -1], axis=-1)
        legal = ((scale > 0.) & (progress[ix] @ progress[ix].T >= MIN_COS * scale)
                 & (dist <= .5 * (travel[:, None] + travel[None, :])))
        np.fill_diagonal(legal, False)
        use = legal.any(axis=1)
        peers[ix[use]] = ix[np.where(legal, dist, np.inf).argmin(axis=1)[use]]
    return peers


def fit_weight(predictor, target):
    num = (predictor * target).sum(axis=(1, 2))
    den = (predictor ** 2).sum(axis=(1, 2)) + RIDGE * (target ** 2).sum(axis=(1, 2))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0.).clip(0., 1.5)


def prefix_check(accel, peers):
    """Compare lag-one predictions on the final two already observed steps."""
    donor = np.where(peers >= 0, peers, np.arange(len(peers)))
    target = accel[:, 1:]
    lag = (accel[:, :-1], accel[donor, :-1])
    cut = FIT_PAIRS
    err = []
    for x in lag:
        b = fit_weight(x[:, :cut], target[:, :cut])
        err.append(((b[:, None, None] * x[:, cut:] - target[:, cut:]) ** 2).sum(axis=(1, 2)))
    zero = (target[:, cut:] ** 2).sum(axis=(1, 2))
    accept = (peers >= 0) & (err[1] + 1e-10 * zero < np.minimum(err[0], zero))
    return accept, donor


def candidates(history, query, cv):
    if (history.shape != (len(query), 8, 2) or cv.shape != (len(query), 12, 2)
            or not np.isfinite(history).all() or not np.isfinite(cv).all()):
        raise ValueError('Require finite observed histories and CV forecasts')
    peers = peers_in_prefix(history, query)
    a = np.diff(history, n=2, axis=1)
    support, donor = prefix_check(a, peers)
    source = (a, a[donor])
    impulse = np.stack([fit_weight(z[:, :-1], a[:, 1:])[:, None] * z[:, -1] for z in source], axis=1)

    step = np.arange(1., 13.)
    decay = np.exp2(-step / HALF_LIFE)
    # A different time constant preserves a distinct shape even on straight paths.
    kernel = np.exp2(-step / IMPULSE_HALF_LIFE).cumsum().cumsum()
    centered = history - history[:, -1:]
    t = np.arange(8.) - 3.5
    slow = np.einsum('t,ntd->nd', t, centered) / (t @ t)
    base = slow[:, None] * step[None, :, None] + (-centered[:, -2] - slow)[:, None] * decay.cumsum()[None, :, None]
    delta = base[:, None] + impulse[:, :, None] * kernel[None, None, :, None] - cv[:, None]
    peak = np.linalg.norm(delta, axis=-1).max(axis=-1)
    common = np.where(support, peak.min(axis=1), 0.)
    ratio = np.divide(common[:, None], peak, out=np.zeros_like(peak), where=peak > 0.)
    result = cv[:, None] + delta * ratio[..., None, None]
    return result, dict(neighbor_rows=int((peers >= 0).sum()), prefix_pass_rows=int(support.sum()),
                        nonzero_cap_rows=int((common > 0.).sum()))
