"""Match frozen CV-relative edits without reselecting recipients."""

import numpy as np

from src.evaluation.m3w_step_budget_match import match

VARIANTS = ('original', 'smaller_peak', 'matched_steps')


def build(cv, control, joint, query):
    """Preserve zero-edit queries and bind a common roster for all variants."""
    if (cv.ndim != 3 or cv.shape[1:] != (12, 2) or cv.shape != control.shape or cv.shape != joint.shape
            or query.shape != (len(cv),) or not np.issubdtype(query.dtype, np.integer)
            or any(not np.isfinite(p).all() for p in (cv, control, joint))):
        raise ValueError('Expected aligned frozen trajectories and query ids')
    moved = np.stack([np.any(p != cv, axis=(1, 2)) for p in (control, joint)])
    pairs = {k: np.stack((control, joint)).copy() for k in VARIANTS}
    keep = np.ones(len(cv), bool)
    counts = dict(eligible_queries=0, collapsed_queries=0, one_edit_queries=0,
                  two_edit_queries=0, original_edits=0, retained_edits=0)
    peak, path = [], []
    for q in np.unique(query):
        rows = np.flatnonzero(query == q)
        active = [rows[moved[a, rows]] for a in range(2)]
        n = len(active[0])
        if len(active[1]) != n or n > 2 or n > len(rows)-1:
            raise ValueError('Frozen edit count is not matched within the query')
        if n == 0:
            continue
        counts['original_edits'] += n
        counts['eligible_queries'] += 1
        counts['one_edit_queries' if n == 1 else 'two_edit_queries'] += 1
        idx = np.stack(active)
        delta = pairs['original'][np.arange(2)[:, None], idx]-cv[idx]
        # Repeating one recipient does not change the minimum over trajectories.
        got = match(np.repeat(delta, 2, axis=1) if n == 1 else delta)
        if got is None:
            keep[rows] = False
            counts['collapsed_queries'] += 1
            continue
        paths, profile, scale = got
        for k in VARIANTS[1:]:
            pairs[k][np.arange(2)[:, None], idx] = cv[idx]+paths[k][:, :n]
        counts['retained_edits'] += n
        peak.append(scale)
        path.append(float(2*n*profile.sum()/np.linalg.norm(delta, axis=-1).sum()))
    for p in pairs.values():
        actual = (p[:, keep] != cv[keep]).any(axis=(2, 3))
        if not np.array_equal(actual, moved[:, keep]):
            raise ValueError('Step matching changed an edited recipient')
    return dict(keep=keep, pairs=pairs, counts=counts,
                peak_retained=float(np.mean(peak)) if peak else None,
                path_retained=float(np.mean(path)) if path else None)
