"""Donor-only coordinate perturbation and fixed-coverage action comparison."""

import numpy as np


MIN_POWER = -6
NOISE_SD = .5
KEEP_NATIVE = .5


def transform(h, y, valid, factor, jitter=None):
    if (h.ndim != 3 or h.shape[1:] != (8, 2) or y.shape != (len(h), 12, 2)
            or valid.shape != y.shape[:2] or not 0. < factor <= 1.):
        raise ValueError("Require aligned 8/12 donor arrays and a positive scale")
    origin = h[:, -1].mean(0)
    x = factor*(h-origin)
    if jitter is not None:
        if jitter.shape != h.shape or not np.isfinite(jitter).all():
            raise ValueError("Invalid observed jitter")
        x = x+jitter
    label = factor*(y-origin)-x[:, -1:]
    return x, np.where(valid[..., None], label, 0.)


def match_gain(gain, legal, count):
    """Match the actual edit count, even when a new score is negative."""
    if (gain.shape != legal.shape or not np.isfinite(gain).all() or legal.dtype != bool
            or not 0 <= count <= legal.any(1).sum()):
        raise ValueError("Invalid score or insufficient legal recipients")
    a = np.where(legal, gain, -np.inf)
    choice = a.argmax(1)
    order = np.argsort(-a[np.arange(len(a)), choice], kind="stable")[:count]
    out = np.zeros(len(a), dtype=np.uint8)
    out[order] = choice[order]+1
    return out
