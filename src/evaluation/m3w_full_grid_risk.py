"""Original row-risk replay and exact all-edited free-completion sensitivity."""

import numpy as np


def assess(cv, proposal, target, valid):
    if (cv.shape != proposal.shape or cv.shape != target.shape or cv.shape != (2, 12, 2)
            or valid.shape != (2, 12) or valid.dtype != bool
            or not all(np.isfinite(x).all() for x in (cv, proposal, target[valid]))):
        raise ValueError('Expected two frozen edited tracks and an aligned twelve-step mask')
    c = np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.)
    p = np.where(valid, np.linalg.norm(proposal-target, axis=-1), 0.)
    distance = np.linalg.norm(proposal-cv, axis=-1)
    if np.any(distance.max(1) <= 0.):
        raise ValueError('Expected two nonidentity edits')
    observed = valid.sum(1)
    row_cv = c.sum(1)/np.maximum(observed, 1)
    row_harm = np.maximum((p-c).sum(1)/np.maximum(observed, 1), 0.)
    original = float((row_harm.sum()+distance.max(1)[observed == 0].sum())/row_cv.sum()) if row_cv.sum() > 0 else None
    floor = c.sum()
    upper_harm = np.maximum((p-c).sum(1)+(distance*~valid).sum(1), 0.).sum()
    return dict(original_all=original, full_grid_upper=float(upper_harm/floor) if floor > 0. else None,
                partial_tracks=int(((0 < observed) & (observed < 12)).sum()), unknown_tracks=int((observed == 0).sum()))
