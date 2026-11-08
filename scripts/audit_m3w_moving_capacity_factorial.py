"""Factor candidate trajectories and per-query edit budgets on Moving A."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
from scripts import audit_m3w_moving_support_bridge as previous
from src.world_model import m3w_moving_capacity_factorial as model

study, run, h = previous.study, previous.run, previous.h
OUT = study.OUT.parent/"capacity_factorial_v1"
CAPS = (1, 2, 4, 8, None)
BANKS = ("original", "expanded")


def analyze(m, rec, ix, groups, y, valid, cut):
    readout = run.parent.readout
    cv, old = m.floor[ix], m.bank[ix]
    ols = old[:, run.NAMES.index(run.parent.OLS)]
    ols8 = old[:, run.NAMES.index("history_ols8_velocity")]
    add = model.extra_bank(m.d["history"][ix], cv, old, ols, ols8)
    target, mask, q = y[ix], valid[ix], m.d["queries"][ix]
    w = 1./m.d["crowd"][ix]
    norm = readout.comparison_masses(ols, cv, cv, target, mask, w)
    offset = norm["known_delta"]-norm["unknown_radius"]
    strong = norm["known_delta"]+norm["unknown_radius"] < 0.
    errors = readout.errors
    e = errors(cv, target, mask)[0]
    known = mask.any(1)
    easy = known & (e > 0.) & (e <= cut["easy"])
    count = Counter(views=1, ols_certified_better=int(strong))
    detail = {}
    for name, bank in zip(BANKS, (old, np.concatenate((old, add), axis=1)), strict=True):
        a = np.column_stack([errors(p, target, mask)[0] for p in bank.swapaxes(0, 1)])
        k = a.argmin(1)
        gain = np.where(known, w*(e-a[np.arange(len(e)), k]), 0.)
        for cap in CAPS:
            key = f"{name}_{cap if cap is not None else 'all'}"
            ix0 = model.plan(gain, easy, groups, cap)
            value = float(gain[ix0].sum())
            hit = len(ix0) > 0 and offset+value > 0.
            count[key] = int(hit)
            count["strong_"+key] = int(hit and strong)
            count["edits_"+key] = int(len(ix0)) if hit else 0
            count["new_paths_"+key] = int((k[ix0] >= old.shape[1]).sum()) if hit else 0
            detail[key] = value
            if not hit:
                continue
            p = cv.copy()
            p[ix0] = bank[ix0, k[ix0]]
            result = readout.score_recording({run.parent.CV: cv, "ols": ols, "candidate": p},
                {"strong": ("ols", "candidate", run.parent.CV)}, target, mask, q, rec,
                easy_cut=cut["easy"], hard_cut=cut["hard"])
            risk = result["metrics"]["candidate"]["risk"]
            if (not risk["supported"] or risk["selected"] != len(ix0)
                    or readout.merge_comparisons([result["comparisons"]["strong"]])["completion_lower_pp"] <= 0.):
                raise ValueError("Factorial witness disagrees with unchanged scoring")
    for cap in CAPS:
        k = str(cap) if cap is not None else "all"
        if count["expanded_"+k] < count["original_"+k]:
            raise ValueError("Expanded menu must contain original actions")
    if strong:
        denom = norm["reference"]
        for key in detail:
            detail[key] = 100.*(offset+detail[key])/denom
    return count, detail if strong else {}


def main():
    torch.set_num_threads(4)
    study.register()
    p = study.OUT/"bridge.json"
    prev = run.read(p)
    if prev["registration"] != run.sha(study.OUT/"bridge_registration.json") or prev["confirmation_access"]:
        raise ValueError("Require the previous single-query bridge")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in (p, study.OUT/"bridge_registration.json", study.OUT/"analysis.json")],
        extra_candidates=model.NAMES, per_query_caps=CAPS, no_recording_cap=True,
        radius="new paths clipped to row's original maximum CV disagreement",
        exact_in_class="known individually CV-improving edits with at least one easy reference, distinct rows",
        original_2pct=True, confirmation_access=False, outcome_aware_only=True,
        not_upper_bound_for_general_2pct_policy=True)))
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    parents, controls, ranks = h.cost.banks()
    cuts = run.cuts_by_source()
    result = {s: [] for s in prev["by_site"]}
    gaps = []
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        for s in result:
            z = [analyze(m, rec, ix, group, y, valid, cuts[s])
                 for rec, (ix, group) in records.items() if str(d["sites"][ix[0]]) == s]
            count = sum((c for c, _ in z), Counter())
            old = next(r["counts"] for r in prev["by_site"][s] if r["seed"] == m.seed)
            if count["original_1"] != old.get("bridge_all", 0):
                raise ValueError("Original per-query-one exact capacity changed")
            result[s].append(dict(seed=m.seed, counts=count))
            gaps.extend(gap for _, gap in z if gap)
        print("per-query capacity checked", m.seed, flush=True)
        del m
    total = sum((z["counts"] for rows in result.values() for z in rows), Counter())
    if len(gaps) != total["ols_certified_better"]:
        raise ValueError("Strong baseline cohort changed")
    summary = dict(counts=total, strong_margin_pp={k: dict(median=float(np.median([r[k] for r in gaps])),
                   worst=float(min(r[k] for r in gaps))) for k in gaps[0]})
    run.once(OUT/"analysis.json", run.stable(dict(registration=run.sha(f), by_site=result, summary=summary,
        original_2pct=True, confirmation_access=False, outcome_aware_only=True)))
    print(run.stable(summary), flush=True)


if __name__ == "__main__":
    main()
