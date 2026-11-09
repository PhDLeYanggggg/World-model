"""Paired risk selection at fixed edit slots, individually or per query."""
import itertools

import numpy as np


MARGINS = (0., .005, .015, .035)
PENALTIES = (.5, 1., 3.)


def choose(prediction, query, scale, rule, joint):
    out = np.zeros(len(prediction), bool)
    if rule is None:
        return out
    margin, penalty = rule
    utility = prediction[:, 0]-margin
    if not joint:
        return utility-penalty*np.maximum(prediction[:, 1:], 0.).mean(1) > 0.
    order = np.argsort(query, kind='stable')
    offsets = np.r_[0, np.unique(query, return_counts=True)[1].cumsum()]
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        at = order[lo:hi]
        if len(at) > 2:
            raise ValueError('Expected at most two fixed slots in a query')
        best = 0.
        for mask in itertools.product((False, True), repeat=len(at)):
            ix = at[np.asarray(mask)]
            value = utility[ix]@scale[ix] - penalty*np.maximum(
                scale[ix]@prediction[ix, 1:], 0.).mean()
            if value > best:
                best = value
                out[at] = mask
    return out


def calibrate(cells, kind):
    guarded = kind != 'gain'
    penalties = PENALTIES if guarded else (0.,)
    grid = [None, *itertools.product(MARGINS, penalties)]
    table = []
    for rule in grid:
        parts = []
        for v in cells.values():
            seed_results = []
            for pred, q, scale, delta, mass in v:
                select = choose(pred, q, scale, rule, kind == 'joint')
                seed_results.append(np.r_[100.*delta[select].sum(0)/mass, select.mean() if len(select) else 0.])
            parts.append(np.mean(seed_results, axis=0))
        gain, a, e, s, rate = np.mean(parts, axis=0)
        table.append(dict(rule=rule, gain=float(gain), risk=[float(a), float(e), float(s)],
                          rate=float(rate), admissible=not guarded or max(a, e, s) <= 1e-12))
    best = max((v for v in table if v['admissible']), key=lambda v: (v['gain'], -v['rate']))
    return best['rule'], table
