"""Outcome-aware fixed-roster capacity and exact missing-step attribution."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments

REFERENCES = ("cv", "ols4", "hindcast")
SUBSETS = ("complete", "partial", "absent")
POSITIVE_PP = 1e-6


def mask_parts(ref, pred, cv, y, valid):
    known, lower = paired(ref, pred, y, valid)
    mass = float(np.linalg.norm(cv-y, axis=-1)[valid].sum())
    if not mass > 0.:
        raise ValueError("Require observed CV reference mass")
    any_valid = valid.any(1)
    changed = (ref != pred).any((1, 2))
    out = {}
    for k, mask in zip(SUBSETS, (valid.all(1), any_valid & ~valid.all(1), ~any_valid), strict=True):
        out[k] = dict(rows=int(mask.sum()), changed=int((mask & changed).sum()),
                      observed=float(100*known[mask].sum()/mass),
                      debit=float(100*(known[mask]-lower[mask]).sum()/mass),
                      lower=float(100*lower[mask].sum()/mass))
    return out


def coefficients(cv, ols, bank, incumbent, active, y, valid, easy_cut):
    refs = (cv, ols, bank[:, incumbent])
    on = np.flatnonzero(active)
    m = bank.shape[1]
    floor = cv[on]
    target, mask = y[on], valid[on]
    risk = [moments(floor, p, target, mask, easy_cut) for p in bank[on].swapaxes(0, 1)]
    fields = ("all_excess", "easy_excess", "easy_signed")
    c = np.stack([np.column_stack([v[k] for v in risk]).ravel() for k in fields])
    offsets = [paired(r[~active], cv[~active], y[~active], valid[~active])[1].sum() for r in refs]
    gains = [np.column_stack([paired(r[on], p, target, mask)[1].sum(1) for p in bank[on].swapaxes(0, 1)]).ravel()
             for r in refs]
    mass = float(np.linalg.norm(cv-y, axis=-1)[valid].sum())
    full_easy_mass = float(moments(cv, cv, y, valid, easy_cut)["ref_easy"].sum())
    return dict(rows=on, arms=m, risk=c, easy_cap=BUDGET*full_easy_mass, easy=bool(risk and risk[0]["easy"].any()),
                offsets=np.asarray(offsets), gain=np.asarray(gains), mass=mass)


def solve(c, seconds=3.):
    """Maximize the smallest lower gain to all three fixed references."""
    n, m = len(c["rows"]), c["arms"]
    if not n or not c["easy"] or not c["mass"] > 0.:
        return None, dict(status="no_support", upper_pp=None)
    columns = n*m
    one = coo_array((np.ones(columns), (np.repeat(np.arange(n), m), np.arange(columns))), shape=(n, columns+1))
    unit = np.abs(c["risk"]).max(1).clip(1.)
    risk = csc_array(np.c_[c["risk"]/unit[:, None], np.zeros(3)])
    gains = csc_array(np.c_[-c["gain"]/c["mass"], np.ones(3)])
    mat = vstack((one, risk, gains), format="csc")
    lo = np.r_[np.ones(n), np.full(6, -np.inf)]
    hi = np.r_[np.ones(n), 0., 0., c["easy_cap"]/unit[2], c["offsets"]/c["mass"]]
    z = milp(np.r_[np.zeros(columns), -1.], integrality=np.r_[np.ones(columns), 0.],
             bounds=Bounds(np.r_[np.zeros(columns), -np.inf], np.r_[np.ones(columns), np.inf]),
             constraints=LinearConstraint(mat, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    b = getattr(z, "mip_dual_bound", None)
    upper = float(-100*b) if b is not None and np.isfinite(b) else None
    receipt = dict(status={0: "optimal", 1: "unresolved", 2: "risk_infeasible"}.get(z.status, "unresolved"),
                   upper_pp=upper)
    if z.x is None:
        return None, receipt
    use = z.x[:-1].reshape(n, m) > .5
    if not np.all(use.sum(1) == 1):
        raise ValueError("Solver changed fixed roster")
    arm = use.argmax(1)
    receipt["objective_pp"] = float(100*z.x[-1])
    return arm, receipt
