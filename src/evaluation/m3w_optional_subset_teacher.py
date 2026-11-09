"""Exact donor-only optional-edit labels, including forced inclusion."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from src.evaluation.m3w_exact_slot_headroom import lower
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model.m3w_causal_risk_moments import excess, target

TOL = 1e-8


def build(cv, ols, path, y, valid, q, cut):
    chosen = np.flatnonzero(np.linalg.norm(path-cv, axis=-1).max(1) > 0.)
    if not len(chosen):
        return None
    w = 1./np.bincount(q)[q]
    mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    if mass <= 0.:
        return dict(chosen=chosen, no_reference=True)
    v = target(cv, path[:, None], y, valid, cut[0])[:, 0]
    picked = v[chosen]
    base = lower(ols, cv, y, valid)
    g = np.stack((w*lower(cv, path, y, valid), w*(lower(ols, path, y, valid)-base)))[:, chosen]
    scale = 100./mass
    constraints, _ = excess(picked, 0.)
    a = np.vstack((np.column_stack((constraints.T, np.zeros(3))),
                   np.r_[-(picked[:, 4] > 0.).astype(float), 0.],
                   np.column_stack((-g*scale, np.ones(2)))))
    ub = np.r_[0., 0., .02*v[:, 4].sum(), -1., 0., scale*(w@base)]
    return dict(chosen=chosen, a=a, ub=ub, local=g[0]*scale, easy=picked[:, 4] > 0.,
                cv=cv, ols=ols, path=path, y=y, valid=valid, cut=cut, weight=w, scale=scale)


def solve(d, force=None):
    n = len(d["chosen"])
    lo = np.r_[np.zeros(n), -np.inf]
    if force is not None:
        lo[force] = 1.
    r = milp(np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
             bounds=Bounds(lo, np.r_[np.ones(n), np.inf]),
             constraints=LinearConstraint(d["a"], -np.inf, d["ub"]),
             options=dict(time_limit=2., mip_rel_gap=0.))
    if r.status == 2:
        return None, "infeasible"
    if r.status != 0 or r.x is None:
        return None, "unresolved"
    mask = r.x[:-1] > .5
    if force is not None and not mask[force]:
        raise ValueError("Forced-in teacher lost its required edit")
    p = d["cv"].copy()
    idx = d["chosen"][mask]
    p[idx] = d["path"][idx]
    pscore = policy(p, d["cv"], d["cv"], d["y"], d["valid"], ["r"]*len(p),
                    easy_cut=d["cut"][0], hard_cut=d["cut"][1])["recordings"]["r"]
    actual = min(d["weight"] @ lower(k, p, d["y"], d["valid"]) for k in (d["cv"], d["ols"]))*d["scale"]
    np.testing.assert_allclose(actual, r.x[-1], atol=1e-7, rtol=1e-7)
    if not pscore["supported"]:
        return None, "numerical_boundary"
    return (mask, float(actual)), "optimal"


def labels(d):
    """NaN means unknown, not a negative teacher example."""
    n = len(d["chosen"])
    out = np.zeros((n, 3), dtype=np.float32)
    if d.get("no_reference"):
        out[:, 0] = np.nan
        return out, {"no_reference": 1}
    out[:, 0] = d["local"] > TOL
    if not d["easy"].any():
        return out, {"no_easy": 1}
    optimum, status = solve(d)
    stats = {status: 1}
    if optimum is None:
        if status != "infeasible":
            out[:, 1:] = np.nan
        return out, stats
    keep, gain = optimum
    if gain <= TOL:
        return out, {"nonpositive": 1}
    out[keep, 1:] = 1.
    for i in np.flatnonzero(~keep):
        forced, code = solve(d, int(i))
        stats[f"forced_{code}"] = stats.get(f"forced_{code}", 0) + 1
        if forced is not None:
            out[i, 2] = forced[1] > TOL
        elif code != "infeasible":
            out[i, 2] = np.nan
    return out, stats
