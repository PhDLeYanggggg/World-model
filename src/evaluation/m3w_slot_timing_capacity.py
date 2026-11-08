"""Hindsight timing capacity at exact per-recording count and radius multiset."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.evaluation.m3w_selection_forecast_readout import policy
from src.evaluation.m3w_streaming_menu_capacity import POSITIVE_PP, respects_tokens, token_rows

POS = "verified_positive_both"


def assess(cv, ols, y, valid, query, eligible, cap, rows, slots, n_slots, paths, cuts, seconds=2.):
    if n_slots == 0:
        return "zero_edits"
    rows, slots = np.asarray(rows), np.asarray(slots)
    easy = moments(cv, cv, y, valid, cuts[0])["ref_easy"]
    if not easy.any():
        return "absent_easy"
    if not easy[rows].any():
        return "no_eligible_easy"
    if not np.array_equal(np.unique(slots), np.arange(n_slots)):
        raise ValueError("Timing alternatives dropped an original slot")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    n, nq = len(rows), len(cap)
    group = lambda ids, size: coo_array((np.ones(n), (ids, np.arange(n))), shape=(size, n+1))
    by_query = group(query[rows], nq)
    offset = paired(ols, cv, y, valid)[1].sum(1)
    gain = np.stack([paired(ref[rows], paths, y[rows], valid[rows])[1].sum(1) for ref in (cv, ols)])
    gain[1] -= offset[rows]
    moments_ = moments(cv[rows], paths, y[rows], valid[rows], cuts[0])
    risk = np.stack([moments_[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    scale = np.abs(risk).max(1).clip(1.)
    matrix = vstack((group(rows, len(cv)), group(slots, n_slots), by_query,
        csc_array(np.column_stack((risk/scale[:, None], np.zeros(3)))),
        csc_array(np.r_[easy[rows] > 0., 0.][None]),
        csc_array(np.column_stack((-gain/mass, np.ones(2))))), format="csc")
    lo = np.r_[np.full(len(cv), -np.inf), np.ones(n_slots), np.full(nq+3, -np.inf), 1., -np.inf, -np.inf]
    hi = np.r_[np.ones(len(cv)+n_slots), cap, 0., 0., .02*easy.sum()/scale[2], np.inf, 0., offset.sum()/mass]
    tokens, upper = token_rows(by_query, eligible)
    matrix = vstack((hstack((matrix, csc_array((matrix.shape[0], nq)))), tokens), format="csc")
    z = milp(c=np.r_[np.zeros(n), -1., np.zeros(nq)], integrality=np.r_[np.ones(n), np.zeros(nq+1)],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf, np.zeros(nq)], np.r_[np.ones(n), np.inf, np.full(nq, 10.)]),
        constraints=LinearConstraint(matrix, np.r_[lo, np.full(2*nq, -np.inf)], np.r_[hi, upper]),
        options=dict(time_limit=seconds, mip_rel_gap=0.))
    if z.x is not None:
        take = z.x[:n] > .5
        count = np.bincount(query[rows[take]], minlength=nq)
        if (len(np.unique(rows[take])) != n_slots or len(np.unique(slots[take])) != n_slots
                or take.sum() != n_slots or (count > cap).any() or not respects_tokens(count, eligible)):
            raise ValueError("Timing witness violates exact slot or streaming constraints")
        pred = cv.copy()
        pred[rows[take]] = paths[take]
        rec = policy(pred, cv, cv, y, valid, np.zeros(len(cv)), easy_cut=cuts[0], hard_cut=cuts[1])["recordings"]["0.0"]
        actual = 100.*np.array([paired(ref, pred, y, valid)[1].sum() for ref in (cv, ols)])/mass
        np.testing.assert_allclose(actual, 100.*(gain[:, take].sum(1)+[0., offset.sum()])/mass, atol=1e-9, rtol=1e-8)
        if rec["supported"] and actual.min() > POSITIVE_PP:
            return POS
    if z.status == 2:
        return "risk_infeasible"
    dual = getattr(z, "mip_dual_bound", None)
    if dual is not None and -100.*dual <= POSITIVE_PP:
        return "no_strict_positive_bound"
    return "unresolved"
