"""Angle choice with a fixed incumbent and separate utility components."""

import numpy as np


def choices(logits, displacement, roster):
    if (logits.shape[-2:] != (3, 6) or displacement.shape != logits.shape[:-1]
            or roster.shape != displacement.shape[:-1] or not np.isfinite(logits).all()
            or not np.isfinite(displacement).all() or (displacement <= 0.).any()
            or (roster < 2).any()):
        raise ValueError('Invalid causal shortlist')
    p = 1./(1.+np.exp(-logits[..., :5].clip(-80., 80.)))
    trajectory = displacement*(p[..., 3]-p[..., 4])
    pair = 2.*displacement/roster[..., None]*np.tanh(logits[..., 5])
    t = (trajectory-trajectory[:, :1])/displacement.max(1, keepdims=True)
    r = (pair-pair[:, :1])/(2.*displacement/roster[:, None]).max(1, keepdims=True)
    legal = (t > 0.) & (r > 0.) & (p[..., 1] >= p[:, :1, 1])
    score = np.where(legal, np.minimum(t, r), 0.)
    return t.argmax(1), score.argmax(1), score.max(1)


def threshold(score, accept):
    """Complete equal-score groups; no unsafe replacement on calibration."""
    if score.ndim != 1 or score.shape != accept.shape or not np.isfinite(score).all():
        raise ValueError('Invalid calibration arrays')
    floor, count = None, 0
    for s in sorted(set(score[score > 0.]), reverse=True):
        group = score == s
        if not accept[group].all():
            break
        floor, count = float(s), count+int(group.sum())
    return floor, count
