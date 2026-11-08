"""Two causal slots with separate utility and reference-support roles."""

import numpy as np

from src.world_model.m3w_opendd_reserve import EASY_THRESHOLD

GAIN_SCALE = .25
SUPPORT_SCALE = .10


def choose(packets):
    """Choose gain, support, and a count/time-matched gain control."""
    gain_row, support, matched = None, None, None
    for rows, gain, easy in packets:
        if not (len(rows) == len(gain) == len(easy)):
            raise ValueError("Misaligned query scores")
        if not np.isfinite(gain).all() or not np.isfinite(easy).all():
            raise ValueError("Nonfinite query scores")
        if gain_row is None and len(rows) and gain.max() > 0.:
            gain_row = int(rows[gain.argmax()])
        eligible = rows != gain_row if gain_row is not None else np.ones(len(rows), dtype=bool)
        if support is None and (eligible & (easy >= EASY_THRESHOLD)).any():
            support = int(rows[np.where(eligible, easy, -np.inf).argmax()])
            matched = int(rows[np.where(eligible, gain, -np.inf).argmax()])
        if gain_row is not None and support is not None:
            break
    return gain_row, support, matched
