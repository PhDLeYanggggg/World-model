"""Measure how much zero-harm non-easy editing can bridge the OLS4 gap."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
from scripts import audit_m3w_moving_sparse_support as previous
from src.world_model import m3w_moving_support_bridge as model

study, run, h, OUT = previous.study, previous.run, previous.h, previous.OUT
CAPS = (*previous.CAPS, None)


def bridge(m, rec, case, ols, y, valid, cut):
    ix, offset = case["ix"], case["offset"]
    cv, bank, target, mask = m.floor[ix], m.bank[ix], y[ix], valid[ix]
    q = m.d["queries"][ix]
    w = 1./m.d["crowd"][ix]
    readout = run.parent.readout
    m0 = readout.comparison_masses(ols[ix], cv, cv, target, mask, w)
    strong = m0["known_delta"]+m0["unknown_radius"] < 0.
    gain, support, all_pick, easy_pick = [], [], [], []
    for rows, kinds in case["qs"]:
        z = w[rows]*case["benefit"][rows, kinds]
        a = study.best(rows, kinds, z, (z > 0.) & mask[rows].any(1))
        b = study.best(rows, kinds, z, case["labels"][rows, kinds, 0])
        all_pick.append(a)
        easy_pick.append(b)
        gain.append(0. if a is None else w[a[0]]*case["benefit"][a])
        support.append(0. if b is None else w[b[0]]*case["benefit"][b])
    gain, support = np.asarray(gain), np.asarray(support)
    counts = Counter()
    for cap in CAPS:
        value, pivot, rest = model.best_bridge(gain, support, cap)
        hit = pivot is not None and offset+value > 0.
        tag = str(cap) if cap is not None else "all"
        counts["bridge_"+tag] = int(hit)
        counts["strong_bridge_"+tag] = int(hit and strong)
        if not hit:
            continue
        p = cv.copy()
        plan = [easy_pick[pivot], *(all_pick[i] for i in rest)]
        for a, b in plan:
            p[a] = bank[a, b]
        z = readout.score_recording({run.parent.CV: cv, "witness": p, "ols": ols[ix]},
            {"strong": ("ols", "witness", run.parent.CV)}, target, mask, q, rec,
            easy_cut=cut["easy"], hard_cut=cut["hard"])
        if (not z["metrics"]["witness"]["risk"]["supported"]
                or z["metrics"]["witness"]["risk"]["selected"] != len(plan)
                or readout.merge_comparisons([z["comparisons"]["strong"]])["completion_lower_pp"] <= 0.):
            raise ValueError("Zero-harm bridge failed exact original scorer")
    if counts["bridge_1"] != int(case["labels"][..., 1].any()):
        raise ValueError("One-edit exact bridge must match the sparse oracle")
    return counts


def main():
    previous.torch.set_num_threads(4)
    study.register()
    a, b, c = (OUT/f"{k}.json" for k in ("analysis", "capacity_registration", "capacity"))
    audit = run.read(c)
    if audit["registration"] != run.sha(b) or run.read(b)["parents"][0]["sha256"] != run.sha(a):
        raise ValueError("Require verified sparse budget audit")
    f = OUT/"bridge_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(p) for p in (a, b, c)], original_2pct=True, confirmation_access=False,
        outcome_aware_only=True, exact_in_class="one known easy positive anchor; one positive known edit per query",
        not_upper_bound_for_general_2pct_policy=True, cap=list(CAPS))))
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    parents, controls, ranks = h.cost.banks()
    cuts = run.cuts_by_source()
    results = {s: [] for s in audit["by_site"]}
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        ols, cases = study.prepare(m, records, y, valid)
        for s in results:
            total = sum((bridge(m, rec, r, ols, y, valid, cuts[s]) for rec, r in cases.items() if r["site"] == s), Counter())
            past = next(z["counts"] for z in audit["by_site"][s] if z["seed"] == m.seed)
            if total["bridge_all"] < past.get("safe_class_unlimited", 0):
                raise ValueError("Non-easy options must not reduce available capacity")
            results[s].append(dict(seed=m.seed, counts=total))
        print("bridge exact witnesses checked", m.seed, flush=True)
        del m
    total = sum((z["counts"] for rows in results.values() for z in rows), Counter())
    run.once(OUT/"bridge.json", run.stable(dict(registration=run.sha(f), by_site=results, summary=total,
        original_2pct=True, confirmation_access=False, outcome_aware_only=True)))
    print(run.stable(total), flush=True)


if __name__ == "__main__":
    main()
