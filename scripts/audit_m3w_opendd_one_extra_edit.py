"""Check whether one additional edit could repair frozen support failures."""

from collections import Counter
from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_record_teacher as teacher

study, v, run, two, hard, m = teacher.study, teacher.v, teacher.run, teacher.two, teacher.hard, teacher.m
OUT = teacher.OUT
SCALE = (1., .5, .25)
GROUPS = ("no_credits", "no_reachable_easy")


def assess(cv, ols, path, y, mask, base, cut):
    selected = (base != cv).any((1, 2))
    legal = ~selected & (path != cv).any((1, 2))
    prior = two.moments(cv, base, y, mask, cut)
    easy = legal & prior["easy"]
    base_gain = np.array([v.paired(k, base, y, mask)[1].sum() for k in (cv, ols)])
    gain_ref = np.column_stack([v.paired(k, cv, y, mask)[1].sum(1) for k in (cv, ols)])
    keys = ("all_excess", "easy_excess", "easy_signed")
    past = np.array([prior[k][selected].sum() for k in keys])
    cap = [0., 0., .02*prior["ref_easy"].sum()]
    any_risk, any_joint = np.zeros(len(cv), dtype=bool), np.zeros(len(cv), dtype=bool)
    values = {}
    for alpha in SCALE:
        p = path if alpha == 1. else cv+alpha*(path-cv)
        now = two.moments(cv, p, y, mask, cut)
        risk = np.column_stack([now[k] for k in keys])
        passed = easy & ((p != cv).any((1, 2))) & ((risk+past) <= cap).all(1)
        added = np.column_stack([v.paired(k, p, y, mask)[1].sum(1) for k in (cv, ols)])
        margin = base_gain+added-gain_ref
        joint = passed & (margin > 0.).all(1)
        any_risk |= passed
        any_joint |= joint
        winner = int(np.argmax(np.where(joint, margin.min(1), -np.inf))) if joint.any() else None
        values[str(alpha)] = (winner, p)
    category = ("qualified" if any_joint.any() else "not_dual_positive" if any_risk.any() else
                "risk_infeasible" if easy.any() else "no_legal_easy")
    return category, values


@torch.inference_mode()
def evaluate(d, bank, path, starts, release, clock, cuts):
    cv, ols = bank[:2]
    y, mask, rec = (d[k] for k in ("y", "valid", "rec"))
    e = two.moments(cv, cv, y, mask, cuts[0])["easy"]
    log = {k: Counter() for k in GROUPS}
    for r, seq in study.packets(d, cv, ols, path, starts, release):
        plan, _ = m.stream(iter(seq), lambda x: clock(torch.from_numpy(x)).numpy(), "adaptive_queue")
        used = np.concatenate([z[0] for z in plan])
        if len(used):
            has_easy = any(e[p.rows[j]].any() for p, (z, _, _) in zip(seq, plan, strict=True)
                           if len(z) for j in np.flatnonzero(p.size == len(z)))
            if has_easy:
                continue
        kind = "no_reachable_easy" if len(used) else "no_credits"
        at = np.flatnonzero(rec == r)
        base = cv[at].copy()
        base[np.searchsorted(at, used)] = path[used]
        status, opts = assess(cv[at], ols[at], path[at], y[at], mask[at], base, cuts[0])
        log[kind].update(total=1, **{status: 1})
        for a, (row, p) in opts.items():
            if row is None:
                continue
            pred = base.copy()
            pred[row] = p[row]
            check = v.parent.score(dict(y=y[at], valid=mask[at], rec=rec[at]), cv[at], ols[at], pred, cuts)
            if check["dual_positive"] != 1 or check["edits"] != len(used)+1:
                raise ValueError("One-edit witness failed the original scorer")
            log[kind][f"qualifies_at_{a}"] += 1
    return log


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, _, _, clock, _, _ = teacher.register()
    f, p = OUT/"support_registration.json", OUT/"analysis.json"
    a = run.read(p)
    earlier = study.OUT/"clock_capacity.json"
    if a["registration"] != key or not a["original_2pct"] or a["confirmation_access"]:
        raise ValueError("Require the completed recording teacher experiment")
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p), run.artifact(earlier)],
        original_2pct=True, confirmation_access=False, outcome_aware_only=True, changes_main_budget=False,
        frozen_clock_groups=GROUPS, amplitudes=SCALE, allowed_extra_edits=1)))
    old_cap = run.read(earlier)["by_site"]
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    result = {}
    for s, (d, _, starts) in sites.items():
        bank = v.parent.prior.analytic(d)
        cuts = np.quantile(np.concatenate([z[0]["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for item in old["by_site"][s]:
            seed = item["seed"]
            release = two.fit.load(outer[f"{s}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(item["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            first = next(p for _, seq in study.packets(d, *bank[:2], path, starts, release) for p in seq)
            n = first.features.shape[1]+m.QUEUE_WIDTH
            net = two.fit.load(clock[f"{s}_{seed}_full"], lambda: v.model.network(n), study.STEPS)
            cell = evaluate(d, bank, path, starts, release, net, cuts)
            old_view = next(z["capacity"] for z in old_cap[s] if z["seed"] == seed)
            if any(cell[k]["total"] != old_view.get(k, 0) for k in GROUPS):
                raise ValueError("The frozen missing-support groups changed")
            result[s].append(dict(seed=seed, groups=cell))
        print(s, "support ceiling complete", flush=True)
    totals = {k: sum((r["groups"][k] for z in result.values() for r in z), Counter()) for k in GROUPS}
    run.once(OUT/"support_capacity.json", run.stable(dict(registration=run.sha(f), by_site=result, summary=totals,
        outcome_aware_only=True, original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
