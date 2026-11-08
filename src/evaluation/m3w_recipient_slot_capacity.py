"""Label-aware recipient capacity at fixed query, radius and shape rule."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_selection_forecast_readout import policy
from src.evaluation.m3w_streaming_menu_capacity import POSITIVE_PP


def assess(cv, ols, y, valid, rows, slots, paths, cuts, seconds=2.):
    if not len(rows):
        return "zero_edits"
    m = moments(cv[rows], paths, y[rows], valid[rows], cuts[0])
    easy = moments(cv, cv, y, valid, cuts[0])["ref_easy"]
    if not m["easy"].any():
        return "no_eligible_easy" if easy.any() else "absent_easy"
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    if mass <= 0.:
        raise ValueError("Positive easy mass must imply positive total reference mass")
    n = len(rows)
    _, row = np.unique(rows, return_inverse=True)
    _, slot = np.unique(slots, return_inverse=True)
    nr, ns = int(row.max()+1), int(slot.max()+1)
    group = lambda ids, size: coo_array((np.ones(n), (ids, np.arange(n))), shape=(size, n+1))
    offset = paired(ols, cv, y, valid)[1].sum(1)
    gain = np.stack([paired(ref[rows], paths, y[rows], valid[rows])[1].sum(1) for ref in (cv, ols)])
    gain[1] -= offset[rows]
    risk = np.stack([m[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    scale = np.abs(risk).max(1).clip(1.)
    matrix = vstack((group(row, nr), group(slot, ns),
        csc_array(np.column_stack((risk/scale[:, None], np.zeros(3)))),
        csc_array(np.r_[m["easy"], 0.][None]),
        csc_array(np.column_stack((-gain/mass, np.ones(2))))), format="csc")
    lo = np.r_[np.full(nr, -np.inf), np.ones(ns), np.full(3, -np.inf), 1., -np.inf, -np.inf]
    hi = np.r_[np.ones(nr+ns), 0., 0., .02*easy.sum()/scale[2], np.inf, 0., offset.sum()/mass]
    z = milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(matrix, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    if z.x is not None:
        take = z.x[:n] > .5
        if len(set(rows[take])) != ns or len(set(slots[take])) != ns or take.sum() != ns:
            raise ValueError("Solver violated frozen slots or assigned a recipient twice")
        pred = cv.copy()
        pred[rows[take]] = paths[take]
        rec = policy(pred, cv, cv, y, valid, np.zeros(len(cv)), easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]["0.0"]
        actual = 100.*np.array([paired(ref, pred, y, valid)[1].sum() for ref in (cv, ols)])/mass
        np.testing.assert_allclose(actual, 100.*(gain[:, take].sum(1)+[0., offset.sum()])/mass, atol=1e-9, rtol=1e-8)
        if rec["supported"] and actual.min() > POSITIVE_PP:
            return "verified_positive_both"
    if z.status == 2:
        return "risk_infeasible"
    dual = getattr(z, "mip_dual_bound", None)
    if dual is not None and -100.*dual <= POSITIVE_PP:
        return "no_strict_positive_bound"
    return "unresolved"
