"""Separate strong-baseline deficits from choice errors in pedestrian recordings."""

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
from scripts import run_m3w_moving_sparse_support as study

OUT, run, h = study.OUT, study.run, study.h
CAPS = (1, 2, 4, 8)


def capacity(m, rec, r, ols, y, valid, cut):
    ix, labels, offset = r["ix"], r["labels"], r["offset"]
    cv, b = m.floor[ix], m.bank[ix]
    weights = 1./m.d["crowd"][ix]
    target, mask, q = y[ix], valid[ix], m.d["queries"][ix]
    readout = run.parent.readout
    rel = readout.comparison_masses(ols[ix], cv, cv, target, mask, weights)
    strong = rel["known_delta"]+rel["unknown_radius"] < 0.
    c = Counter(views=1, ols_known_better=int(rel["known_delta"] < 0.),
                ols_certified_better=int(strong), cv_certified_better=int(offset > 0.),
                cv_any=int(labels[..., 0].any()), both_any=int(labels[..., 1].any()),
                both_when_ols_certified_better=int(strong and labels[..., 1].any()))
    pick = []
    for i, k in r["qs"]:
        z = r["benefit"][i, k]*weights[i]
        a = study.best(i, k, z, labels[i, k, 0])
        if a is not None:
            row, family = a
            pick.append((float(r["benefit"][row, family]*weights[row]), row, family))
    pick.sort(reverse=True)
    gains = np.asarray([z[0] for z in pick])
    feasible = np.flatnonzero(offset+np.cumsum(gains) > 0.)
    count = int(feasible[0])+1 if len(feasible) else None
    if (count == 1) != bool(labels[..., 1].any()):
        raise ValueError("Sparse capacity disagrees with exact one-edit oracle")
    for k in CAPS:
        c[f"safe_class_at_most_{k}"] = int(count is not None and count <= k)
        c[f"strong_at_most_{k}"] = int(strong and count is not None and count <= k)
    c["safe_class_unlimited"] = int(count is not None)
    c["strong_unlimited"] = int(strong and count is not None)
    if count is not None:
        p = cv.copy()
        for _, i, k in pick[:count]:
            p[i] = b[i, k]
        metrics = readout.score_recording({run.parent.CV: cv, "witness": p, "ols": ols[ix]},
            {"strong": ("ols", "witness", run.parent.CV)}, target, mask, q, rec,
            easy_cut=cut["easy"], hard_cut=cut["hard"])
        if (not metrics["metrics"]["witness"]["risk"]["supported"]
                or metrics["metrics"]["witness"]["risk"]["selected"] != count
                or readout.merge_comparisons([metrics["comparisons"]["strong"]])["completion_lower_pp"] <= 0.):
            raise ValueError("Exact safe-class budget witness failed original scoring")
    return c


def main():
    torch.set_num_threads(4)
    key = study.register()
    a, heads = (OUT/f"{k}.json" for k in ("analysis", "head_freeze"))
    previous = run.read(a)
    if previous["registration"] != key or previous["heads"] != run.sha(heads):
        raise ValueError("Require the frozen sparse study")
    f = OUT/"capacity_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p) for p in (a, heads)],
        budget_caps=CAPS, original_2pct=True, confirmation_access=False, outcome_aware_only=True,
        exact_in_class="at most one individually CV-improving, known-easy edit per query; sorted weighted gains",
        not_upper_bound_for_general_2pct_policy=True)))
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    parents, controls, ranks = h.cost.banks()
    cuts = run.cuts_by_source()
    results = {s: [] for s in previous["by_site"]}
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        ols, cases = study.prepare(m, records, y, valid)
        for s in results:
            c = sum((capacity(m, rec, r, ols, y, valid, cuts[s]) for rec, r in cases.items() if r["site"] == s), Counter())
            old = next(z for z in previous["by_site"][s] if z["seed"] == m.seed)
            if c["both_any"] != old["metrics"]["window_dual_oracle"]["dual_positive"]:
                raise ValueError("Cross-source capacity did not reconcile")
            results[s].append(dict(seed=m.seed, counts=c))
        print("safe-class budgets checked", m.seed, flush=True)
        del m
    total = sum((z["counts"] for rows in results.values() for z in rows), Counter())
    run.once(OUT/"capacity.json", run.stable(dict(registration=run.sha(f), by_site=results, summary=total,
        original_2pct=True, confirmation_access=False, outcome_aware_only=True)))
    print(run.stable(total), flush=True)


if __name__ == "__main__":
    main()
