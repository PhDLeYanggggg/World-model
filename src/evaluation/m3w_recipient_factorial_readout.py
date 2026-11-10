"""Score a locked recipient/path factorial from caller-supplied targets."""

import numpy as np

from src.evaluation.m3w_pneuma_matched_readout import contrast
from src.evaluation.m3w_selection_forecast_readout import policy
from src.evaluation.m3w_songdo_matched_readout import absolute, coverage

CELLS = tuple(a+b for a in 'IAJ' for b in 'IAJ')


def restore(cv, off, packet):
    q = packet['query_index']
    if (not np.array_equal(off, packet['offsets']) or q.ndim != 1
            or not np.issubdtype(q.dtype, np.integer) or np.any(np.diff(q) <= 0)
            or np.any((q < 0) | (q >= len(off)-1)) or np.any(np.diff(off)[q] < 3)):
        raise ValueError('Invalid frozen roster')
    radius = packet['radius'] if len(q) else np.empty(0)
    if radius.shape != q.shape or not np.all(np.isfinite(radius) & (radius > 0.)):
        raise ValueError('Invalid frozen edit radii')
    pred, sets = {}, {}
    for a in CELLS:
        p = cv.copy()
        if len(q):
            row, path = packet[a+'_rows'], packet[a+'_paths']
            if (row.shape != (len(q), 2) or not np.issubdtype(row.dtype, np.integer)
                    or np.any(row[:, 0] == row[:, 1])
                    or not np.all((off[q, None] <= row) & (row < off[q+1, None]))
                    or path.shape != (len(q), 2, 12, 2) or not np.isfinite(path).all()):
                raise ValueError('Invalid frozen edit')
            ids = np.sort(row, axis=1)
            known = sets.setdefault(a[0], ids)
            if not np.array_equal(known, ids):
                raise ValueError('Path rule changed its assigned recipients')
            if not np.allclose(np.linalg.norm(path-cv[row], axis=-1).max(-1), radius[:, None], atol=1e-9, rtol=1e-9):
                raise ValueError('Path rule changed its radius')
            p[row.ravel()] = path.reshape(-1, 12, 2)
        pred[a] = p
    return q, pred


def score(cv, ols, y, valid, offsets, packet, contrasts, *, easy_cut, hard_cut):
    if (cv.ndim != 3 or cv.shape[1:] != (12, 2) or cv.shape != ols.shape or cv.shape != y.shape
            or not np.isfinite(cv).all() or not np.isfinite(ols).all() or valid.dtype != bool
            or valid.shape != cv.shape[:2] or not np.isfinite(y[valid]).all()
            or offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer) or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != len(cv) or np.any(np.diff(offsets) < 2)
            or not 0 < easy_cut <= hard_cut < np.inf):
        raise ValueError('Invalid aligned inputs or fixed risk cutoffs')
    q, pred = restore(cv, offsets, packet)
    full, counts = coverage(valid, offsets)
    recording = np.repeat('member', len(cv))
    arms = {}
    for a, p in dict(cv=cv, ols4=ols, **pred).items():
        r = absolute(p, y, valid, full)
        if a in pred:
            row = policy(p, cv, cv, y, valid, recording, easy_cut=easy_cut, hard_cut=hard_cut)
            r.update(original_risk=row['recordings']['member'],
                     intervention_rate=float(np.any(p != cv, axis=(1, 2)).mean()))
        arms[a] = r
    paired = None
    if len(q):
        take = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in q])
        compact = np.r_[0, np.cumsum(np.diff(offsets)[q])]
        weight = np.repeat(1./np.diff(compact), np.diff(compact))
        yy, vv, cc = y[take], valid[take], cv[take]
        mass = float(weight @ np.where(vv, np.linalg.norm(cc-yy, axis=-1), 0.).sum(1))
        paired = {k: contrast(pred[a][take], pred[b][take], yy, vv, weight, mass, compact)
                  for k, (a, b) in contrasts.items()}
    return dict(coverage=counts, arms=arms, matched=len(q), unmatched=len(offsets)-1-len(q), paired=paired)
