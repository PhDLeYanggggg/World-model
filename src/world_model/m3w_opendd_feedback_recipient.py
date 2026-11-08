"""Resolved forecast errors for recipient choice, without current-query labels."""

import numpy as np

from src.world_model.m3w_ols_replacement import rollout


SCOPES = ("availability", "aligned", "wrong_agent")


def traces(history, ids, offsets, recording, starts, stride, observe):
    """Read only the previous disjoint cohort after its 12 steps have elapsed."""
    if history.shape != (len(ids), 8, 2) or stride <= 0:
        raise ValueError("Expected the registered eight-step observations")
    aligned = np.zeros((len(ids), 7), dtype=np.float32)
    wrong = np.zeros_like(aligned)
    cv, ols, _, scale = rollout(history, 12)
    previous = None
    latest = {}
    exchanged = 0
    for q in np.lexsort((starts, recording)):
        lo, hi = offsets[q:q+2]
        now = starts[q]
        if previous is not None and recording[previous] == recording[q]:
            last = starts[previous] + 19*stride
            if last >= now:
                raise ValueError("Feedback may not overlap the current observation window")
            at = np.arange(offsets[previous], offsets[previous+1])
            xy = observe(at, last, now)
            if xy.shape != (len(at), 12, 2):
                raise ValueError("Unaligned past observations")
            visible = np.isfinite(xy).all(2)
            for i, row in enumerate(at):
                on = visible[i]
                if not on.any():
                    continue
                target = xy[i, on]
                a, b = (np.linalg.norm(p[row, on]-target, axis=1).mean()/scale[row] for p in (cv, ols))
                latest[int(ids[row])] = (last, on.mean(), a, b)
        else:
            latest.clear()

        if len(np.unique(ids[lo:hi])) != hi-lo:
            raise ValueError("Agent identities must be unique inside a query")
        for row in range(lo, hi):
            v = latest.get(int(ids[row]))
            if v is not None:
                when, fraction, a, b = v
                aligned[row] = (1., np.log1p((now-when)/stride), fraction, np.log1p(a), np.log1p(b),
                                np.arcsinh(a-b), a > b)
        wrong[lo:hi] = aligned[lo:hi]
        good = lo+np.flatnonzero(aligned[lo:hi, 0])
        if len(good) > 1:
            good = good[np.argsort(ids[good])]
            wrong[good, 3:] = aligned[np.roll(good, 1), 3:]
            exchanged += len(good)
        previous = q
    base = aligned.copy()
    base[:, 3:] = 0.
    return dict(availability=base, aligned=aligned, wrong_agent=wrong), exchanged
