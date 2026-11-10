"""Stream distinct-recipient candidate pairs in the existing exact tie order."""

import numpy as np


def blocks(actor, size=4096):
    actor = np.asarray(actor)
    if actor.ndim != 1 or type(size) is not int or size < 1:
        raise ValueError('Expected one-dimensional actor roster and positive block size')
    a, b = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64)
    used = 0
    for i in range(len(actor)-1):
        j = np.flatnonzero(actor[i+1:] != actor[i])+i+1
        pos = 0
        while pos < len(j):
            n = min(size-used, len(j)-pos)
            a[used:used+n] = i
            b[used:used+n] = j[pos:pos+n]
            used += n
            pos += n
            if used == size:
                yield a.copy(), b.copy()
                used = 0
    if used:
        yield a[:used].copy(), b[:used].copy()


def update_best(current, left, right, scores):
    """Retain the first exact maximizer, including ties across blocks."""
    for arm, v in scores.items():
        if v.shape != left.shape or np.isnan(v).any() or np.isposinf(v).any():
            raise ValueError('Unexpected matched candidate score')
        best = int(v.argmax())
        score = float(v[best])
        if np.isfinite(score) and (arm not in current or score > current[arm][1]):
            current[arm] = (np.array([left[best], right[best]]), score)
