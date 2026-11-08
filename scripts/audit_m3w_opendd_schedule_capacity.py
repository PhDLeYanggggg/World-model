"""Separate risk and accuracy feasibility within the frozen two-role bank."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_support_schedule as schedule

run, v, two, hard, base, record = schedule.run, schedule.v, schedule.two, schedule.hard, schedule.base, schedule.record
OUT = schedule.OUT
FLAGS = ("risk", "cv", "ols", "dual", "risk_cv", "risk_ols", "joint")


def exact(big, small, ids, ref_easy, ols_offset):
    found = {k: False for k in FLAGS}

    def add(z, first=None):
        g, risk, easy, all_mass = z
        cv = g[:, 0]+(0. if first is None else first[0][0]) > 0.
        ols = g[:, 1]+ols_offset+(0. if first is None else first[0][1]) > 0.
        safe, _ = schedule.model.support_targets(g, risk, easy, all_mass, ref_easy, first, ols_offset)
        for k, q in zip(FLAGS, (safe, cv, ols, cv & ols, safe & cv, safe & ols, safe & cv & ols), strict=True):
            found[k] |= bool(q.any())

    subset = lambda z, ix: tuple(x[ix] for x in z)
    add(subset(big, ids))
    add(subset(small, ids))
    for j in ids:
        if all(found.values()):
            break
        add(subset(small, ids[ids != j]), tuple(x[j] for x in big))
    return found


def evaluate(d, bank, path, cut, ref):
    cv, ols = bank[:2]
    big = cv+base.model.GAIN_SCALE*(path-cv)
    small = cv+base.model.SUPPORT_SCALE*(path-cv)
    y, valid, rec = d["y"], d["valid"], d["rec"]
    a, b = (record.contributions(cv, ols, p, y, valid, cut) for p in (big, small))
    legal = (small != cv).any((1, 2))
    count, reasons = Counter(), Counter()
    for r in np.unique(rec):
        at = np.flatnonzero(rec == r)
        ids = np.flatnonzero(legal[at])
        c = exact(tuple(z[at] for z in a[:4]), tuple(z[at] for z in b[:4]), ids,
                  b[2][at].sum(), b[4][at].sum())
        count.update(c)
        if c["joint"]:
            reason = "joint"
        elif b[2][at].sum() <= 0.:
            reason = "no_full_easy_reference"
        elif not c["risk"]:
            reason = "easy_exists_but_no_risk_feasible_action"
        elif not c["risk_cv"] and not c["risk_ols"]:
            reason = "risk_feasible_neither_reference"
        elif not c["risk_cv"]:
            reason = "risk_feasible_beats_ols_only"
        elif not c["risk_ols"]:
            reason = "risk_feasible_beats_cv_only"
        else:
            reason = "risk_feasible_beats_each_with_different_actions"
        reasons[reason] += 1
    if (count["joint"] != ref["any_up_to_two"] or reasons["no_full_easy_reference"] != ref["no_full_easy_reference"]
            or sum(reasons.values()) != len(np.unique(rec))):
        raise ValueError("Feasibility partition changed the verified roster")
    return dict(counts=count, reasons=reasons)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, _, old, source, *_ = schedule.register()
    p = OUT/"analysis.json"
    a = run.read(p)
    if a["registration"] != key or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require verified whole-clock results")
    f = OUT/"capacity_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(p),
        original_2pct=True, confirmation_access=False, outcome_aware_only=True,
        matching="one or both legal roles, any distinct recipient or time, original amplitudes; no edits not counted",
        diagnostic="separate risk-only, one-reference and two-reference feasibility and their intersection")))
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    rows = {}
    for s, d in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), .25)
        rows[s] = []
        for z in old["by_site"][s]:
            with np.load(run.checked(z["prediction"]), allow_pickle=False) as data:
                path = data["comoving_ego"]
            ref = next(k["capacity"] for k in a["by_site"][s] if k["seed"] == z["seed"])
            rows[s].append(dict(seed=z["seed"], **evaluate(d, bank, path, cut, ref)))
        print(s, "feasibility partition audited", flush=True)
    flat = [r for group in rows.values() for r in group]
    summary = {k: sum((r[k] for r in flat), Counter()) for k in ("counts", "reasons")}
    run.once(OUT/"capacity_breakdown.json", run.stable(dict(registration=run.sha(f),
        summary=summary, by_site=rows, original_2pct=True, confirmation_access=False, outcome_aware_only=True)))


if __name__ == "__main__":
    main()
