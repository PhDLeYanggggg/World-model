"""Prefix-only cohorts and paired observed/missing-label forecast comparisons."""
import numpy as np


def cohorts(visible, stride=12):
    if visible.ndim != 2 or visible.dtype != bool or stride < 1:
        raise ValueError('Expected boolean agent-by-frame support and positive stride')
    n = visible.shape[1] - 19 * stride
    if n <= 0 or len(visible) < 2:
        return
    supported = visible[:, :n].copy()
    for i in range(1, 8):
        supported &= visible[:, i * stride:i * stride + n]
    next_free = 0
    for start in np.flatnonzero(supported.sum(0) >= 2):
        if start >= next_free:
            yield int(start), np.flatnonzero(supported[:, start])
            next_free = start + 19 * stride + 1


def geometry(past, offsets, scale):
    """Nearest peer chosen by current separation; no source or global position."""
    result = np.zeros((len(past), 6), dtype=np.float32)
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        p = past[lo:hi]
        if len(p) < 2:
            continue
        v = (p[:, -1] - p[:, -4]) / 3.
        d = p[None, :, -1] - p[:, None, -1]
        dist = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(dist, np.inf)
        j = dist.argmin(1)
        i = np.arange(len(p))
        r, speed = d[i, j], v[j] - v
        s = scale[lo:hi]
        radial = (r * speed).sum(1) / np.linalg.norm(r, axis=1).clip(1e-8)
        past_d = np.linalg.norm(p[j] - p, axis=2) / s[:, None]
        result[lo:hi] = np.column_stack((np.log1p(dist[i, j] / s), radial / s,
            np.linalg.norm(speed, axis=1) / s, np.log1p(past_d.mean(1)),
            np.log1p(past_d.min(1)), np.full(len(p), np.log1p(len(p) - 1))))
    return result.clip(-16., 16.)


def choose(score, legal, offsets):
    """At most one edit per query; all arms share the same causal support."""
    if score.shape != legal.shape or not np.isfinite(score).all():
        raise ValueError('Expected finite aligned decision scores')
    result = np.full(len(score), -1, dtype=np.int16)
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        ok = legal[lo:hi]
        if ok.any():
            row, option = np.unravel_index(np.where(ok, score[lo:hi], -np.inf).argmax(), ok.shape)
            result[lo + row] = option
    return result


def paired(ref, pred, label, valid):
    ref_error = np.linalg.norm(ref - label, axis=-1)
    pred_error = np.linalg.norm(pred - label, axis=-1)
    gain = np.where(valid, ref_error - pred_error, 0.)
    debit = np.where(valid, 0., np.linalg.norm(ref - pred, axis=-1))
    return gain, gain - debit
