"""Advance the first intervention credit, then repay it from causal accrual."""

import numpy as np


def decode(scores, legal, recordings, starts, *, early_allowed):
    early_allowed = np.asarray(early_allowed, dtype=bool)
    if early_allowed.shape != (len(scores),):
        raise ValueError("Need one current-query gate per score row")
    out = np.zeros(len(scores), dtype=int)
    previous, closed = None, set()
    for i, (value, allowed, recording, start, gate) in enumerate(
            zip(scores, legal, recordings, starts, early_allowed, strict=True)):
        if recording != previous:
            if recording in closed:
                raise ValueError("Noncontiguous recording")
            closed.add(recording)
            previous, balance, used, last = recording, 0, False, -np.inf
        if start <= last:
            raise ValueError("Non-increasing query time")
        last = start
        # Fifths of an edit: one arrives per eligible query, ten can be stored.
        balance = min(10, balance + int(allowed[1:].any()))
        affordable = np.arange(3)*5 <= max(0, balance)
        if gate and not used:
            affordable[1] = True
        out[i] = np.where(allowed & affordable, value, -np.inf).argmax()
        used |= out[i] > 0
        balance -= 5*out[i]
    return out
