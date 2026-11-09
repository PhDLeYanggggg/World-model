"""Check a frozen Songdo index against already-authorized coordinate rows."""

import numpy as np


def displacements(prefix, index, source, *, stride=12, tick_ms=1001./30.):
    """Join by original CSV row and validate actor, drone and clock identity."""
    history, offsets = prefix['history'], prefix['offsets']
    rows, valid, origin = index['future_index'], index['future_mask'], index['origin']
    counts = np.diff(offsets)
    n = len(history)
    if (history.shape != (n, 8, 2) or rows.shape != (n, 12) or valid.shape != rows.shape
            or not np.issubdtype(rows.dtype, np.integer) or valid.dtype != bool
            or not np.array_equal(valid, rows >= 0) or np.any(rows[~valid] != -1)
            or not len(offsets) or offsets[0] != 0 or offsets[-1] != n or np.any(counts < 2)
            or origin.shape != (len(counts), 2) or prefix['past_ms'].shape != (n, 8)
            or prefix['agent_id'].shape != (n,) or prefix['drone'].shape != (len(counts),)
            or not np.isfinite(history).all() or not np.isfinite(origin).all()
            or not np.isfinite(prefix['past_ms']).all() or not (stride > 0 and np.isfinite(tick_ms) and tick_ms > 0)):
        raise ValueError('Invalid prefix or frozen label index')
    keys = rows[valid]
    original = np.asarray(source['row'])
    xy = np.asarray(source['xy'])
    m = len(keys)
    if (original.shape != (m,) or not np.issubdtype(original.dtype, np.integer)
            or xy.shape != (m, 2) or not np.isfinite(xy).all()
            or any(np.shape(source[k]) != (m,) for k in ('agent', 'drone', 'ms', 'visible'))
            or not np.isfinite(source['ms']).all()
            or len(np.unique(keys)) != m):
        raise ValueError('Invalid or repeated future source rows')
    order = np.argsort(original)
    match = np.argsort(np.argsort(keys))
    use = order[match]
    if not np.array_equal(original[use], keys):
        raise ValueError('Future source row IDs differ from the frozen index')
    actor, step = np.nonzero(valid)
    last_ms = prefix['past_ms'][actor, -1]
    expected = (step+1)*stride*tick_ms
    drone = np.repeat(prefix['drone'], counts)
    if (not np.array_equal(source['agent'][use], prefix['agent_id'][actor])
            or not np.array_equal(source['drone'][use], drone[actor])
            or not np.all(source['visible'][use] == 1)
            or np.any(np.abs(source['ms'][use]-last_ms-expected) > .7*tick_ms+1e-6)):
        raise ValueError('Future actor, drone, visibility or clock does not match the prefix')
    current = history[:, -1]+np.repeat(origin, counts, axis=0)
    y = np.full(rows.shape+(2,), np.nan)
    y[valid] = xy[use]-current[actor]
    return y
