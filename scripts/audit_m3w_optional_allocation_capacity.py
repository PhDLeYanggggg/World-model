"""Compare forced and optional counts under the same two-person cap."""

import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_optional_allocation as study
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack
from src.evaluation.m3w_dual_reference_capacity import action_masses, solve
from src.evaluation.m3w_recipient_release_capacity import STATES, evidence

run, np = study.run, study.np
LIMITS = (1., 4.)


def optional(m, q, cap, seconds):
    """Maximize the same two-reference objective with caps, not equality."""
    row = m["row"]
    n = len(row)
    if not n or not (m["easy"] & (cap[q[row]] > 0)).any():
        return None, dict(infeasible=True, upper_gain_mass=None, solver_status=None)
    columns = np.arange(n)
    by_query = coo_array((np.ones(n), (q[row], columns)), shape=(len(cap), n+1))
    by_agent = coo_array((np.ones(n), (row, columns)), shape=(len(q), n+1))
    scale = np.maximum(abs(m["risk"]).max(1), 1.)
    risk = np.column_stack((m["risk"]/scale[:, None], np.zeros(3)))
    utility = np.column_stack((-m["utility"]/m["scale"], np.ones(2)))
    a = vstack((by_query, by_agent, csc_array(risk), csc_array(np.r_[m["easy"], 0.][None]),
                csc_array(utility)), format="csc")
    lo = np.r_[np.full(len(cap)+len(q)+3, -np.inf), 1., -np.inf, -np.inf]
    hi = np.r_[cap, np.ones(len(q)), 0., 0., m["easy_cap"]/scale[2], np.inf, m["offset"]/m["scale"]]
    opt = milp(np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(a, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    bound = getattr(opt, "mip_dual_bound", None)
    return (None if opt.x is None else opt.x[:-1] > .5), dict(solver_status=int(opt.status),
        infeasible=opt.status == 2, upper_gain_mass=float(-bound*m["scale"])
        if bound is not None and np.isfinite(bound) else None)


def analyze(m, cv, ols, bank, y, valid, q, cap, weight, rec, cut, choice, forced):
    result = None
    for seconds in LIMITS:
        mask, result = choice(m, q, cap, seconds)
        if mask is not None:
            row, arm = m["row"][mask], m["arm"][mask]
            if (np.bincount(q[row], minlength=len(cap)) > cap).any():
                raise ValueError("A witness exceeds the causal cap")
            v = evidence(cv, ols, bank, y, valid, q, cap if forced else None, weight, rec, cut, row, arm)
            if v["risk"]["supported"] and min(v["gain_mass"]) > 1e-7*m["scale"]:
                return dict(result, status="verified_both", evidence=v)
        if result["infeasible"]:
            return dict(result, status="risk_infeasible")
        if result["upper_gain_mass"] is not None and result["upper_gain_mass"] < -1e-7*m["scale"]:
            return dict(result, status="negative_upper")
    return dict(result, status="unresolved")


def main():
    key, identity, cut, _ = study.register()
    out = study.OUT / "capacity_registration.json"
    run.once(out, run.stable(dict(parent_registration=key, code=run.artifact(Path(__file__)),
        solver=run.artifact(Path(solve.__code__.co_filename)), scope="original_fitting_only",
        cap="minimum of inherited count and two", both_references=True, solve_limits=LIMITS,
        label_aware_only=True, confirmation_access=False, original_risk_unchanged=True)))
    key = run.sha(out)
    start = time.monotonic()
    rows = []
    with (study.PRIVATE / "capacity.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for rec, g, _, cv, ols, bank, y, valid, q, counts, w in study.job.prior.prior.old.blocks(
                identity, lambda **_: None):
            f = study.PRIVATE / f"capacity_{rec}.json"
            cap = np.minimum(2, counts)
            if not f.exists():
                bank = cv[:, None]+.25*(bank-cv[:, None])
                m = action_masses(cv, ols, bank, y, valid, w, cut["easy"])
                evaluate = lambda fn, forced: analyze(m, cv, ols, bank, y, valid, q, cap, w, rec, cut, fn, forced)
                run.once(f, run.stable(dict(registration=key, recording=rec, source=g, original=int(counts.sum()),
                    capped=int(cap.sum()), forced=evaluate(solve, True), optional=evaluate(optional, False))))
            v = run.read(f)
            if (v["registration"], v["recording"], v["source"], v["capped"]) != (key, rec, g, int(cap.sum())):
                raise ValueError("Stale capacity checkpoint")
            if v["forced"]["status"] == "verified_both" and v["optional"]["status"] not in ("verified_both", "unresolved"):
                raise ValueError("Relaxation excludes an existing witness")
            rows.append(v)
            status = dict(pid=os.getpid(), elapsed=round(time.monotonic()-start, 1), processed=len(rows),
                          recording=rec, forced=v["forced"]["status"], optional=v["optional"]["status"])
            run.kin.prior.source.atomic(study.PRIVATE / "capacity_heartbeat.json", status)
            print(status, flush=True)
    if len(rows) != 24:
        raise ValueError("Require all 24 original fitting recordings")
    counts = {k: {s: sum(r[k]["status"] == s for r in rows) for s in STATES} for k in ("forced", "optional")}
    run.once(study.OUT / "capacity.json", run.stable(dict(registration=key, counts=counts, recordings=rows,
        original_total=sum(r["original"] for r in rows), two_agent_total=sum(r["capped"] for r in rows),
        newly_verified=sum(r["forced"]["status"] != "verified_both" and r["optional"]["status"] == "verified_both"
                           for r in rows), confirmation_access=False, hindsight_only=True)))


if __name__ == "__main__":
    main()
