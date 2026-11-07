"""Current-query activation with separate donor coverage and offline controls."""
import numpy as np


def split(names, held):
    names = list(names)
    i = names.index(held)
    calibration = [names[(i+j) % len(names)] for j in (1, 2, 3)]
    fitting = [s for s in names if s not in (held, *calibration)]
    if not fitting or len(set(calibration)) != 3 or held in calibration:
        raise ValueError('Need disjoint fitting, calibration and held sources')
    return fitting, calibration


def threshold(scores, target):
    """Use only donor scores; never exceed the donor target on ties."""
    if target <= 0:
        return None
    finite = np.sort(scores[np.isfinite(scores)])[::-1]
    if not len(finite):
        return None
    i = min(target, len(finite))-1
    cut = finite[i]
    return float(np.nextafter(cut, np.inf) if i+1 < len(finite) and finite[i+1] == cut else cut)


def causal_mask(scores, cut):
    return np.isfinite(scores) & (scores >= cut) if cut is not None else np.zeros(len(scores), bool)


def matched_mask(scores, old, count, recording):
    """Offline equal-count comparison; requires the entire scoring roster."""
    out = np.zeros(len(scores), dtype=bool)
    for key in np.unique(recording):
        for n in np.unique(count[recording == key]):
            group = (recording == key) & (count == n)
            target = int((old & group).sum())
            eligible = np.flatnonzero(group & np.isfinite(scores))
            if target > len(eligible):
                raise ValueError('Not enough eligible query options to match control')
            ix = np.argsort(-scores[eligible], kind='stable')[:target]
            out[eligible[ix]] = True
    return out
