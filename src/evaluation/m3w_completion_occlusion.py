"""Stress-test completion radii by hiding known coordinates, not missing labels."""
from hashlib import sha256

import numpy as np


LENGTHS = (1, 2, 4, 8, 12)
KEYS = tuple((kind, length, center) for kind, centers in (
    ('internal', ('interpolation',)), ('terminal', ('constant', 'ols4')))
    for length in LENGTHS for center in centers)


def corridor(track, seen, hidden, center):
    """Use only retained observations for the center and median-speed radius."""
    obs = np.flatnonzero(seen)
    ix = np.asarray(hidden)
    if not len(ix) or not len(obs) or seen[ix].any():
        raise ValueError('Hidden steps must be disjoint from retained observations')
    coords = track[obs]
    if not np.isfinite(coords).all():
        raise ValueError('Expected finite retained coordinates')
    insert = np.searchsorted(obs, ix)
    nearest = np.minimum(abs(ix-obs[np.maximum(insert-1, 0)]),
                         abs(ix-obs[np.minimum(insert, len(obs)-1)]))
    steps = np.linalg.norm(np.diff(coords, axis=0)[np.diff(obs) == 1], axis=1)
    scale = max(.25, float(np.median(steps))) if len(steps) else .25
    if center == 'interpolation':
        if np.any((insert == 0) | (insert == len(obs))):
            raise ValueError('Internal gaps need two retained anchors')
        p = np.column_stack([np.interp(ix, obs, z) for z in coords.T])
    elif center in ('constant', 'ols4'):
        if len(obs) < 4 or not (np.diff(obs[-4:]) == 1).all() or ix.min() <= obs[-1]:
            raise ValueError('Terminal gaps need four consecutive preceding observations')
        velocity = np.array([-.3, -.1, .1, .3]) @ (coords[-4:]-coords[-1]) if center == 'ols4' else 0.
        p = coords[-1]+(ix-obs[-1])[:, None]*velocity
    else:
        raise ValueError('Unexpected corridor center')
    return p, scale*np.minimum(nearest, 12)


def pseudo_gaps(track, released, *, key):
    """One reproducible gap of each kind/length per eligible track."""
    if track.shape != (len(released), 2) or released.dtype != bool:
        raise ValueError('Require a unique aligned agent-step ledger')
    known = np.flatnonzero(released)
    if not len(known):
        return {}
    runs = np.split(known, np.flatnonzero(np.diff(known) != 1)+1)
    rows = {}
    for kind, length, center in KEYS:
        if kind == 'internal':
            eligible = np.concatenate([r[4:-length] for r in runs])
            if not len(eligible):
                continue
            index = int.from_bytes(sha256(f'{key}:{length}'.encode()).digest()[:8], 'little') % len(eligible)
            ix = np.arange(eligible[index], eligible[index]+length)
        else:
            if len(runs[-1]) < length+4:
                continue
            ix = runs[-1][-length:]
        seen = released.copy()
        seen[ix] = False
        p, radius = corridor(track, seen, ix, center)
        error = np.abs(track[ix]-p).max(1)
        ratio = error/radius
        rows[kind, length, center] = np.array([1, length, (ratio <= 1.+1e-12).sum(),
            (ratio <= 2.+1e-12).sum(), (ratio <= 1.+1e-12).all(), (ratio <= 2.+1e-12).all(),
            np.linalg.norm(track[ix]-p, axis=1).sum()], dtype=float)
    return rows


def summary(matrix, draws):
    """Summarize equal-weight source proportions; masks can differ by length."""
    eligible = matrix[:, 0] > 0.
    counts = matrix[eligible].sum(0)
    rows = matrix[eligible]
    if len(rows) != len(matrix) or draws.ndim != 2 or draws.shape[1] != len(matrix):
        raise ValueError('A fixed source roster is required for this contrast')
    values = rows[:, 2:]/rows[:, [1, 1, 0, 0, 1]]
    return dict(gaps=int(counts[0]), hidden_steps=int(counts[1]), source_groups=len(rows),
        **{name: dict(point=float(v.mean()), interval95=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
           for name, v in zip(('point_1x','point_2x','gap_1x','gap_2x','mean_center_error_px'), values.T, strict=True)})
