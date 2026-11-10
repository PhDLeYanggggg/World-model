"""Separate matched utility and full-roster risk, without source access."""

import numpy as np

from src.evaluation.m3w_pneuma_matched_readout import ARMS, contrast
from src.evaluation.m3w_selection_forecast_readout import policy
from src.evaluation.m3w_songdo_matched_readout import absolute, coverage


def restore(cv, offsets, proposal):
    q = proposal['query_index']
    n = len(offsets)-1
    if (q.ndim != 1 or not np.issubdtype(q.dtype, np.integer)
            or np.any(np.diff(q) <= 0) or np.any(q < 0) or np.any(q >= n)):
        raise ValueError('Invalid common-query subset')
    if np.any(np.diff(offsets)[q] < 3):
        raise ValueError('Matched proposals require the frozen minimum of three actors')
    compact = np.r_[0, np.cumsum(np.diff(offsets)[q])]
    if len(q) and (not np.array_equal(compact, proposal['offsets'])
                   or proposal['radius'].shape != q.shape
                   or not np.all(np.isfinite(proposal['radius']) & (proposal['radius'] > 0.))):
        raise ValueError('Unexpected common query sizes or radius')
    predictions = {}
    for arm in ARMS:
        pred = cv.copy()
        if len(q):
            row, path = (proposal[f'{arm}_{k}'] for k in ('rows', 'paths'))
            if (row.shape != (len(q), 2) or not np.issubdtype(row.dtype, np.integer)
                    or not np.all((row >= offsets[q, None]) & (row < offsets[q+1, None]))
                    or np.any(row[:, 0] == row[:, 1]) or path.shape != (len(q), 2, 12, 2)
                    or not np.isfinite(path).all()):
                raise ValueError('Original row indices do not match their eligible queries')
            actual = np.linalg.norm(path-cv[row], axis=-1).max(-1)
            if not np.allclose(actual, proposal['radius'][:, None], atol=1e-9, rtol=1e-9):
                raise ValueError('Matched edit radii differ')
            pred[row.ravel()] = path.reshape(-1, 12, 2)
        predictions[arm] = pred
    take = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in q]) if len(q) else np.empty(0, dtype=int)
    return q, take, compact, predictions


def score(cv, ols, median, y, valid, offsets, proposal, *, easy_cut=None, hard_cut=None):
    if (cv.ndim != 3 or cv.shape[1:] != (12, 2) or any(p.shape != cv.shape for p in (ols, median, y))
            or valid.dtype != bool or valid.shape != cv.shape[:2] or not np.isfinite(y[valid]).all()
            or any(not np.isfinite(p).all() for p in (cv, ols, median)) or offsets.ndim != 1
            or not np.issubdtype(offsets.dtype, np.integer) or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != len(cv) or np.any(np.diff(offsets) < 2)
            or ((easy_cut is not None or hard_cut is not None)
                and not (easy_cut is not None and hard_cut is not None and 0 < easy_cut <= hard_cut < np.inf))):
        raise ValueError('Invalid forecasts, target masks, roster or fixed cutoffs')
    q, take, compact, edits = restore(cv, offsets, proposal)
    full, counts = coverage(valid, offsets)
    arms = {}
    for a, p in dict(cv=cv, ols4=ols, median4=median, **edits).items():
        row = absolute(p, y, valid, full)
        row['intervention_rate'] = float(np.any(p != cv, axis=(1, 2)).mean())
        if a in edits and easy_cut is not None:
            check = policy(p, cv, cv, y, valid, np.repeat('recording', len(y)), easy_cut=easy_cut, hard_cut=hard_cut)
            row['original_risk'] = check['recordings']['recording']
            row['easy_known_gain_percent'] = check['subsets']['easy']['gain_percent']
            row['hard_known_gain_percent'] = check['subsets']['hard']['gain_percent']
        arms[a] = row
    paired = None
    if len(q):
        weight = np.repeat(1./np.diff(compact), np.diff(compact))
        yy, ok, base = y[take], valid[take], cv[take]
        mass = float(weight @ np.where(ok, np.linalg.norm(base-yy, axis=-1), 0.).sum(1))
        paired = {a: contrast(edits['joint'][take], edits[a][take], yy, ok, weight, mass, compact)
                  for a in ARMS[:-1]}
    return dict(coverage=counts, matched_queries=len(q), unmatched_queries=len(offsets)-1-len(q),
                matched_target_rows=len(take), full_roster=arms, matched_joint_minus=paired,
                risk_on_full_roster=easy_cut is not None, deployment_certified=False)
