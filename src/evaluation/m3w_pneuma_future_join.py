"""Join a declared prefix to labels. This module never opens a real source."""

import numpy as np

from src.data_unification.m3w_pneuma_prefix import east_north, FUTURE, HISTORY, tracks
from src.data_unification.m3w_pneuma_support import ABSENT


def join(stream, prefix, stride):
    history, offsets = prefix['history'], prefix['offsets']
    origin, ticks, ids = (prefix[k] for k in ('origin', 'past_tick', 'agent_id'))
    if (type(stride) is not int or stride < 1 or history.shape != (len(ids), HISTORY, 2)
            or ticks.shape != (len(ids), HISTORY) or not np.issubdtype(ticks.dtype, np.integer)
            or not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(offsets.dtype, np.integer)
            or offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(ids)
            or np.any(np.diff(offsets) < 2) or origin.shape != (len(offsets)-1, 2)
            or not np.isfinite(history).all() or not np.isfinite(origin).all()
            or not np.all(np.diff(ticks, axis=1) == stride)):
        raise ValueError('Invalid frozen causal roster')
    anchors = np.repeat(origin, np.diff(offsets), axis=0)+history[:, -1]
    wanted = ticks[:, -1:]+stride*np.arange(1, FUTURE+1)
    rows = {}
    for i, a in enumerate(ids):
        rows.setdefault(a, []).append(i)
    y = np.zeros((len(ids), FUTURE, 2))
    valid = np.zeros(y.shape[:2], dtype=bool)
    for agent, _, clock, ix, row in tracks(stream):
        if agent not in rows:
            continue
        i = np.asarray(rows.pop(agent))
        lookup = np.searchsorted(clock, wanted[i])
        safe = np.minimum(lookup, max(0, len(clock)-1))
        if not len(clock):
            raise ValueError('Frozen past actor lost its release clock')
        left, t = np.where((lookup < len(clock)) & (clock[safe] == wanted[i]))
        rr, tt, xy = [], [], []
        for a, b in zip(left, t, strict=True):
            j = 4+6*ix[lookup[a, b]]
            lat, lon = row[j].strip(), row[j+1].strip()
            if lat.lower() in ABSENT or lon.lower() in ABSENT:
                continue
            rr.append(i[a])
            tt.append(b)
            xy.append((float(lat), float(lon)))
        if rr:
            y[rr, tt] = east_north(xy)-anchors[rr]
            valid[rr, tt] = True
    if rows:
        raise ValueError('Frozen past actor absent from target source')
    return y, valid
