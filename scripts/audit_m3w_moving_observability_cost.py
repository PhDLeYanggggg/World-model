"""Attribute the frozen pedestrian comparison at identical query edit counts."""

from pathlib import Path
import platform
import sys

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_moving_observability as study
from src.evaluation import m3w_recording_forecast_comparison as readout

run, h, torch = study.run, study.h, study.torch
OUT = study.OUT / "cost_attribution_v1"
ARMS = ("geometry", "motion", "no_missing", "geometry_no_fill")
PARTS = ("direction", "removed", "added_safe", "added_fill", "direction_fill")


def decision_partition(reference, proposed, core):
    """Partition changed rows without assigning either action's full radius twice."""
    edit, orig = proposed >= 0, reference >= 0
    fill = edit & (core < 0)
    return dict(direction=edit & orig & (reference != proposed) & ~fill,
                removed=orig & ~edit, added_safe=edit & ~orig & ~fill,
                added_fill=fill & ~orig, direction_fill=fill & orig & (reference != proposed))


def paired_terms(a, b, denominator, target, mask, weights, partitions):
    total = readout.comparison_masses(a, b, denominator, target, mask, weights)
    parts = {}
    for k, ix in partitions.items():
        p = readout.comparison_masses(a[ix], b[ix], denominator[ix], target[ix], mask[ix], weights[ix])
        parts[k] = {f: p[f] for f in ("known_delta", "unknown_radius")}
    for f in ("known_delta", "unknown_radius"):
        np.testing.assert_allclose(sum(p[f] for p in parts.values()), total[f], atol=1e-8, rtol=1e-10)
    return total, parts


@torch.inference_mode()
def score(m, ix, groups, y, valid, legal, context, net, frozen):
    c, b, x = m.floor[ix], m.bank[ix], m.x[ix]
    scale, mask, target = m.scale[ix], valid[ix], y[ix]
    avail = legal[ix]
    n, k = avail.shape
    ent = frozen(torch.from_numpy(x.reshape(n*k, -1).astype(np.float32))).numpy().reshape(n, k, 3)
    ent = ent * scale[:, None, None]
    ctx = np.broadcast_to(context[ix, None], (n, k, context.shape[1]))
    cond, pm, pg = net(torch.from_numpy(np.concatenate((x, ctx), axis=-1).reshape(n*k, -1).astype(np.float32)))
    cond = cond.numpy().reshape(n, k, 3) * scale[:, None, None]
    radius = np.linalg.norm(b-c[:, None], axis=-1).max(2)
    costs = dict(no_missing=cond)
    for a, p in (("motion", pm), ("geometry", pg)):
        costs[a] = study.model.costs(cond, p.sigmoid().numpy().reshape(n, k)[:, :1], radius)
    ref = np.full(n, -1, dtype=int)
    pick = {a: ref.copy() for a in ARMS}
    for q in groups:
        ref[q] = study.choose(ent[q], avail[q], study.CAP, False)
        count = int((ref[q] >= 0).sum())
        for a, cost in costs.items():
            pick[a][q] = study.choose(cost[q], avail[q], count, True)
        pick["geometry_no_fill"][q] = study.choose(costs["geometry"][q], avail[q], count, False)
        if any(np.count_nonzero(pick[a][q] >= 0) != count for a in costs):
            raise ValueError("Query-count mismatch")
    weights = 1./m.d["crowd"][ix]
    paths = {a: run.model.selected_paths(v, b, c) for a, v in pick.items()}
    orig = run.model.selected_paths(ref, b, c)
    masses = {a: readout.comparison_masses(orig, paths[a], c, target, mask, weights) for a in ARMS}
    partition = decision_partition(ref, pick["geometry"], pick["geometry_no_fill"])
    _, parts = paired_terms(orig, paths["geometry"], c, target, mask, weights, partition)
    no_fill = pick["geometry_no_fill"]
    filler = (pick["geometry"] >= 0) & (no_fill < 0)
    if not np.array_equal(no_fill[~filler], pick["geometry"][~filler]):
        raise ValueError("Removing filler must preserve admitted choices")
    counts = dict(matched=int((ref >= 0).sum()), filled=int(filler.sum()),
                  filled_unknown=int((filler & ~mask.any(1)).sum()),
                  **{a: int(p.sum()) for a, p in partition.items()})
    return masses, parts, counts


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, _, weights = study.register()
    a, f = (study.OUT / f"{k}.json" for k in ("analysis", "head_freeze"))
    prev, freeze = run.read(a), run.read(f)
    if prev["registration"] != key or prev["heads"] != run.sha(f) or freeze["registration"] != key:
        raise ValueError("Require frozen parents")
    OUT.mkdir(parents=True, exist_ok=True)
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p) for p in (a, f)],
        parts=PARTS, arms=ARMS, original_2pct=True, confirmation_access=False, no_new_training=True,
        primary="exact row partition of matched utility and debit with unchanged full-reference denominator",
        no_fill="drop forced fills only at fixed per-query quota; changed budget, attribution only")))
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    context = study.model.observability_features(d["history"], d["groups"])
    cuts = run.cuts_by_source()
    parents, controls, ranks = h.cost.banks()
    trainer = study.old.study.trainer
    rows = {s: [] for s in prev["by_site"]}
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        _, legal = study.old.cost_targets(m, y, valid, cuts)
        for s in rows:
            name = f"{m.seed}_{s}"
            width = m.x.shape[-1]
            frozen = trainer.load(weights[name], lambda: study.old.model.network(width), study.old.STEPS)
            net = trainer.load(freeze["heads"][name], lambda: study.model.Head(width), study.STEPS)
            scored = [score(m, ix, groups, y, valid, legal, context, net, frozen)
                      for ix, groups in records.values() if str(d["sites"][ix[0]]) == s]
            total = {k: readout.merge_comparisons([r[0][k] for r in scored]) for k in ARMS}
            ref = next(r for r in prev["by_site"][s] if r["seed"] == m.seed)
            effects = {}
            for k, z in total.items():
                if k != "geometry_no_fill":
                    np.testing.assert_allclose(z["completion_lower_pp"], ref["effects"][k+"_vs_entangled"],
                                               atol=1e-12, rtol=0.)
                effects[k] = dict(observed=z["known_pp"], debit=z["known_pp"]-z["completion_lower_pp"],
                                  lower=z["completion_lower_pp"])
            for p in PARTS:
                b = readout.merge_comparisons([{**r[1][p], "reference": r[0]["geometry"]["reference"]} for r in scored])
                effects[p] = dict(observed=b["known_pp"], debit=b["known_pp"]-b["completion_lower_pp"],
                                  lower=b["completion_lower_pp"])
            rows[s].append(dict(seed=m.seed, effects=effects,
                                counts={k: sum(r[2][k] for r in scored) for k in scored[0][2]}))
        print(f"Verified matched actions and costs for seed {m.seed}", flush=True)
        del m
    draw = np.random.default_rng(73107).integers(len(rows), size=(3000, len(rows)))

    def ci(k, f):
        v = np.array([np.mean([r["effects"][k][f] for r in rs]) for rs in rows.values()])
        return dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())

    all_rows = [r for rs in rows.values() for r in rs]
    counts = {k: sum(r["counts"][k] for r in all_rows) for k in all_rows[0]["counts"]}
    if len(all_rows) != 42 or counts["matched"] != prev["summary"]["metrics"]["geometry"]["edits"]:
        raise ValueError("Incomplete cohort")
    summary = {k: {f: ci(k, f) for f in ("observed", "debit", "lower")} for k in (*ARMS, *PARTS)}
    run.once(OUT / "analysis.json", run.stable(dict(registration=run.sha(reg), counts=counts, summary=summary,
        by_site=rows, original_2pct=True, confirmation_access=False, no_new_training=True)))
    print(run.stable(dict(counts=counts, summary=summary)), flush=True)


if __name__ == "__main__":
    main()
