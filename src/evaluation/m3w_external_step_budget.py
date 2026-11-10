"""Match displacement on a frozen pair without changing recipients or directions."""

import numpy as np

from src.evaluation.m3w_pneuma_matched_readout import contrast
from src.evaluation.m3w_step_budget_match import match

VARIANTS = ('original', 'smaller_peak', 'matched_steps')


def build_pair(cv, offsets, queries, proposal, control):
    n = len(queries)
    if (cv.ndim != 3 or cv.shape[1:] != (12, 2) or not np.isfinite(cv).all()
            or offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(cv)
            or not np.issubdtype(offsets.dtype, np.integer) or np.any(np.diff(offsets) < 2)
            or queries.ndim != 1 or not np.issubdtype(queries.dtype, np.integer)
            or np.any(np.diff(queries) <= 0) or np.any((queries < 0) | (queries >= len(offsets)-1))
            or np.any(np.diff(offsets)[queries] < 3)
            or control not in ('independent', 'additive', 'same_family')):
        raise ValueError('Expected the frozen roster and a valid matched subset')
    if not n:
        return dict(queries=queries, rows=np.empty((0, 2, 2), dtype=int), profiles=np.empty((0, 12)),
                    path={k: np.empty((0, 2, 2, 12, 2)) for k in VARIANTS}, collapsed=0)
    arms = (control, 'joint')
    rows = np.stack([proposal[f'{a}_rows'] for a in arms], axis=1)
    path = np.stack([proposal[f'{a}_paths'] for a in arms], axis=1)
    radius = np.asarray(proposal['radius'])
    if (rows.shape != (n, 2, 2) or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows[..., 0] == rows[..., 1]) or (rows < offsets[queries, None, None]).any()
            or (rows >= offsets[queries+1, None, None]).any() or path.shape != (n, 2, 2, 12, 2)
            or not np.isfinite(path).all() or radius.shape != (n,) or not np.all(np.isfinite(radius) & (radius > 0.))):
        raise ValueError('Unbound paths or recipient rows')
    delta = path-cv[rows]
    np.testing.assert_allclose(np.linalg.norm(delta, axis=-1).max(-1),
                               np.broadcast_to(radius[:, None, None], rows.shape), atol=1e-9, rtol=1e-9)
    active, profiles = [], []
    result = {k: [] for k in VARIANTS}
    for i, d in enumerate(delta):
        z = match(d)
        if z is None:
            continue
        values, profile, _ = z
        active.append(i)
        profiles.append(profile)
        for key in VARIANTS:
            result[key].append(path[i] if key == 'original' else cv[rows[i]]+values[key])
    active = np.asarray(active, dtype=int)
    out = dict(queries=queries[active], rows=rows[active], profiles=np.asarray(profiles).reshape(-1, 12),
               path={k: np.asarray(v).reshape(-1, 2, 2, 12, 2) for k, v in result.items()}, collapsed=n-len(active))
    for key in VARIANTS:
        displacement = np.linalg.norm(out['path'][key]-cv[out['rows']], axis=-1)
        if (displacement.max(-1) <= 0.).any():
            raise ValueError('A matched path lost its nonidentity edit')
        if key == 'matched_steps':
            np.testing.assert_allclose(displacement, np.broadcast_to(out['profiles'][:, None, None], displacement.shape),
                                       atol=1e-9, rtol=1e-9)
    return out


def score_pair(cv, offsets, pair, target, valid):
    """Score the secondary utility comparison on a single pre-bound roster."""
    q, rows = pair['queries'], pair['rows']
    if (target.shape != cv.shape or valid.shape != cv.shape[:2] or valid.dtype != bool
            or not np.isfinite(target[valid]).all()):
        raise ValueError('Expected a common target and its original validity mask')
    if not len(q):
        return None
    ids = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in q])
    qoff = np.r_[0, np.cumsum(np.diff(offsets)[q])]
    w = np.repeat(1./np.diff(qoff), np.diff(qoff))
    y, ok, c = target[ids], valid[ids], cv[ids]
    mass = float(w @ np.where(ok, np.linalg.norm(c-y, axis=-1), 0.).sum(-1))
    rel = np.searchsorted(ids, rows)
    if not np.array_equal(ids[rel], rows):
        raise ValueError('An edit is outside the comparison roster')
    out = {}
    for key, path in pair['path'].items():
        control, joint = c.copy(), c.copy()
        control[rel[:, 0]] = path[:, 0]
        joint[rel[:, 1]] = path[:, 1]
        out[key] = contrast(joint, control, y, ok, w, mass, qoff)
    return out
