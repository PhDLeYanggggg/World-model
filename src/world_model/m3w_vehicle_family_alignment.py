"""Replace a training anchor by the learned family when that path is legal."""

import numpy as np


def align_anchor(anchor, rows, bank, legal):
    rows = np.asarray(rows)
    if (rows.shape != (2,) or len(set(rows)) != 2 or bank.shape[:2] != (2, 3)
            or legal.shape != bank.shape[:2]):
        raise ValueError('Expected the same two selected recipients and three families')
    previous = legal & (np.abs(bank-anchor[rows, None]).max(axis=(-1, -2)) <= 1e-10)
    if not previous.any(axis=1).all():
        raise ValueError('Original draw no longer belongs to the legal menu')
    out = anchor.copy()
    use = legal[:, 2]
    out[rows[use]] = bank[use, 2]
    aligned = legal & (np.abs(bank-out[rows, None]).max(axis=(-1, -2)) <= 1e-10)
    if not aligned.any(axis=1).all():
        raise ValueError('Aligned anchor is not a legal path')
    return out, previous, aligned
