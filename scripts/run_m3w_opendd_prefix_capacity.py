"""Bound release timing when each query has a fixed causal recipient rank."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_credit_value as v

OUT = v.OUT
DELAYS = (0, 2)


def solve(release, rows, gain, risk, support, ols_gain, easy_mass, delay):
    if not np.sum(release):
        return "no_credits", None
    if not sum(map(len, rows)):
        raise ValueError("Released credits require ranked candidate rows")
    # Each binary variable is a nonempty prefix, with at most one per query.
    at = np.repeat(np.arange(len(rows)), [len(r) for r in rows])
    sizes = np.concatenate([np.arange(1, len(r)+1) for r in rows])
    prefix = lambda x: np.concatenate([x[r].cumsum(axis=0) for r in rows])
    g, risk, support = prefix(gain), prefix(risk), prefix(support)
    q = np.arange(len(rows))[:, None]
    local = q == at
    past = (q >= at)*sizes
    cum = np.cumsum(release)
    due = np.r_[np.zeros(delay), cum][:len(cum)].copy()
    available = cum-np.r_[0., due[:-1]]
    due[-1] = cum[-1]
    feasible = sizes <= available[at]
    if not (feasible & (support > 0)).any():
        return "no_reachable_easy", None
    zeros = np.zeros(len(q))
    matrix = np.r_[np.c_[local, zeros], np.c_[past, zeros],
                   np.c_[risk.T, np.zeros(3)], np.c_[-g.T, np.ones(2)], np.r_[support, 0.][None]]
    lo = np.r_[zeros, due, [-np.inf]*5, 1.]
    hi = np.r_[np.ones(len(q)), cum, 0., 0., .02*easy_mass, 0., ols_gain, np.inf]
    scale = np.maximum(np.max(np.abs(matrix), axis=1), 1.)
    r = milp(np.r_[np.zeros(len(at)), -1.], integrality=np.r_[np.ones(len(at)), 0.],
             bounds=Bounds(np.r_[np.zeros(len(at)), -np.inf], np.r_[feasible.astype(float), np.inf]),
             constraints=LinearConstraint(matrix/scale[:, None], lo/scale, hi/scale),
             options=dict(time_limit=5., mip_rel_gap=0.))
    if r.status == 2:
        return "risk_infeasible", None
    if r.x is not None:
        on = r.x[:-1] > .5
        spend = np.bincount(at[on], weights=sizes[on], minlength=len(rows))
        if ((local[:, on].sum(1) <= 1).all() and support[on].sum() > 0
                and (g[on].sum(0)+[0., ols_gain]).min() > 0.
                and (risk[on].sum(0) <= [0., 0., .02*easy_mass]).all()
                and v.prior.model.legal_schedule(release, spend, [len(r) for r in rows], delay, True)):
            return "qualified", np.concatenate([r[:int(k)] for r, k in zip(rows, spend, strict=True)])
    bound = getattr(r, "mip_dual_bound", None)
    return ("not_dual_positive" if bound is not None and np.isfinite(bound) and -bound < -1e-8 else "unresolved"), None


@torch.inference_mode()
def evaluate(d, bank, ego, starts, head, cut):
    cv, ols = bank[:2]
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    m = v.two.moments(cv, ego, y, valid, cut[0])
    risk = np.column_stack([m[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    ols_gain = v.paired(ols, cv, y, valid)[1].sum(1)
    gain = np.column_stack((v.paired(cv, ego, y, valid)[1].sum(1),
                            v.paired(ols, ego, y, valid)[1].sum(1)-ols_gain))
    result = {str(k): Counter() for k in DELAYS}
    for r, (rows, p, _) in zip(np.unique(rec), v.packets(d, cv, ols, ego, starts, head), strict=True):
        use = np.flatnonzero(rec == r)
        release = [z[2] for z in p]
        for delay in DELAYS:
            status, choose = solve(release, rows, gain, risk, m["easy"], ols_gain[use].sum(),
                                   m["ref_easy"][use].sum(), delay)
            if status == "qualified":
                pred = cv[use].copy()
                pred[np.searchsorted(use, choose)] = ego[choose]
                score = v.parent.score(dict(y=y[use], valid=valid[use], rec=rec[use]), cv[use], ols[use], pred, cut)
                if score["dual_positive"] != 1 or score["edits"] != sum(release):
                    raise ValueError("Ranked-prefix witness failed the original scorer")
            result[str(delay)][status] += 1
    return result


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, _, heads, old, source, previous, _ = v.register()
    analysis = OUT/"analysis.json"
    values = v.run.read(analysis)
    if values["registration"] != key or values["confirmation_access"]:
        raise ValueError("Require frozen learned value experiment")
    reg = OUT/"prefix_registration.json"
    v.run.once(reg, v.run.stable(dict(code=v.run.artifact(Path(__file__)), parent=v.run.artifact(analysis),
        delays=DELAYS, site_seeds=v.two.study.prior.SEEDS, same_recording_total=True, exact_original_2pct=True,
        comparison="same release clock and local caps; recipient subset must be a prefix of the causal motion rank",
        label_aware_diagnostic=True, confirmation_access=False)))
    sites = {s: v.candidate.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    result = {}
    for s, (d, _, starts) in sites.items():
        cv, ols, *other = v.parent.prior.analytic(d)
        width = v.hard.model.features(d["h"][:1]).shape[1]
        cut = np.quantile(np.concatenate([z[0]["cut_samples"] for k, z in sites.items() if k != s]), [.25, .75])
        result[s] = []
        for r in old["by_site"][s]:
            seed = r["seed"]
            h = v.two.fit.load(heads[f"{s}_{seed}"], lambda: v.hard.model.network(width), v.hard.STEPS)
            with np.load(v.run.checked(r["prediction"]), allow_pickle=False) as z:
                cap = evaluate(d, (cv, ols, *other), z["comoving_ego"], starts, h, cut)
            prev = next(c["capacity"] for c in previous["by_site"][s] if c["seed"] == seed)
            for t, counts in cap.items():
                if counts["qualified"] > prev[t].get("qualified", 0)+prev[t].get("unresolved", 0):
                    raise ValueError("Rank restriction cannot increase exact capacity")
            if cap["0"]["qualified"] > cap["2"]["qualified"]+cap["2"]["unresolved"]:
                raise ValueError("Delay cannot reduce exact capacity")
            result[s].append(dict(seed=seed, counts=cap))
        print(s, "prefix bounds complete", flush=True)
    summary = {str(t): sum((r["counts"][str(t)] for z in result.values() for r in z), Counter()) for t in DELAYS}
    v.run.once(OUT/"prefix_capacity.json", v.run.stable(dict(registration=v.run.sha(reg), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
