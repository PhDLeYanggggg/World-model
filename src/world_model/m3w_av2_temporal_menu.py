"""Reshape when a causal trajectory edit acts, without changing its peak."""

import numpy as np


MODES = ("base", "early", "late", "union")
MIN_WEIGHT = .25


def expanded(cv, bank):
    cv, bank = np.asarray(cv), np.asarray(bank)
    if cv.ndim != 3 or bank.ndim != 4 or (bank.shape[0], *bank.shape[2:]) != cv.shape:
        raise ValueError("Require aligned CV and per-actor forecast banks")
    if not np.isfinite(cv).all() or not np.isfinite(bank).all() or cv.shape[1] < 2:
        raise ValueError("Require finite forecast paths over multiple steps")
    offset = bank-cv[:, None]
    peak = np.linalg.norm(offset, axis=-1).max(-1)
    time = np.linspace(MIN_WEIGHT, 1., cv.shape[1])
    arms = [bank]
    for w in (time[::-1], time):
        shaped = offset*w[None, None, :, None]
        new_peak = np.linalg.norm(shaped, axis=-1).max(-1)
        scale = np.divide(peak, new_peak, out=np.zeros_like(peak), where=new_peak > 0.)
        arms.append(cv[:, None]+shaped*scale[..., None, None])
    return np.concatenate(arms, axis=1)


def membership(columns, count):
    """Original family index is modulo bank width, including the second slot."""
    family = (columns % (3*count)) // count
    groups = {k: (family == i).all(-1) for i, k in enumerate(MODES[:-1])}
    return dict(groups, union=np.ones(len(columns), dtype=bool))
