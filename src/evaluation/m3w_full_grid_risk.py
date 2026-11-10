"""Original row-risk replay and exact full-grid risk sensitivities."""

import numpy as np


def _terms(cv, proposal, target, valid):
    if (cv.shape != proposal.shape or cv.shape != target.shape or cv.shape != (2, 12, 2)
            or valid.shape != (2, 12) or valid.dtype != bool
            or not all(np.isfinite(x).all() for x in (cv, proposal, target[valid]))):
        raise ValueError('Expected two frozen edited tracks and an aligned twelve-step mask')
    c = np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.)
    p = np.where(valid, np.linalg.norm(proposal-target, axis=-1), 0.)
    distance = np.linalg.norm(proposal-cv, axis=-1)
    if np.any(distance.max(1) <= 0.):
        raise ValueError('Expected two nonidentity edits')
    return c, p, distance


def assess(cv, proposal, target, valid):
    c, p, distance = _terms(cv, proposal, target, valid)
    observed = valid.sum(1)
    row_cv = c.sum(1)/np.maximum(observed, 1)
    row_harm = np.maximum((p-c).sum(1)/np.maximum(observed, 1), 0.)
    original = float((row_harm.sum()+distance.max(1)[observed == 0].sum())/row_cv.sum()) if row_cv.sum() > 0 else None
    floor = c.sum()
    upper_harm = np.maximum((p-c).sum(1)+(distance*~valid).sum(1), 0.).sum()
    return dict(original_all=original, full_grid_upper=float(upper_harm/floor) if floor > 0. else None,
                partial_tracks=int(((0 < observed) & (observed < 12)).sum()), unknown_tracks=int((observed == 0).sum()))


def assess_easy(cv, proposal, target, valid, *, easy_cut, limit=.02):
    """Separate a guaranteed easy denominator from worst-case threshold excess.

    The cutoff is an average in native units; the calculations use twelve-step
    sums. Each missing target can vary freely for its scored occurrence.
    """
    if not (np.isfinite(easy_cut) and easy_cut > 0. and np.isfinite(limit) and limit >= 0.):
        raise ValueError('Expected a positive easy cutoff and a nonnegative limit')
    c, p, distance = _terms(cv, proposal, target, valid)
    observed = valid.sum(1)
    row_cv = c.sum(1)/np.maximum(observed, 1)
    observed_easy = (row_cv > 0.) & (row_cv <= easy_cut)
    den = row_cv[observed_easy].sum()
    num = (np.maximum((p-c).sum(1)/np.maximum(observed, 1), 0.)[observed_easy].sum()
           + distance.max(1)[observed == 0].sum())

    floor = c.sum(1)
    complete = observed == 12
    mandatory = complete & (floor > 0.) & (floor/12 <= easy_cut)
    optional = ~complete & (floor/12 <= easy_cut)
    harm_upper = np.maximum((p-c).sum(1)+(distance*~valid).sum(1), 0.)
    excess = harm_upper - limit*floor
    # An incomplete track can be excluded; include it only if it raises excess.
    upper = float(excess[mandatory].sum()+np.maximum(excess[optional], 0.).sum())
    return dict(original_easy=float(num/den) if den > 0. else None,
                easy_support_guaranteed=bool(mandatory.any()),
                optional_easy_tracks=int(optional.sum()),
                easy_excess_supremum=upper,
                selected_easy_pass_for_all=bool(mandatory.any() and upper <= 0.))
