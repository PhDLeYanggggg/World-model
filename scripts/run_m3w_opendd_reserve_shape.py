"""Compare forecast shapes on frozen gain/easy reserve rows and query times."""

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
from scripts import run_m3w_opendd_reserve as prior
from src.world_model import m3w_opendd_reserve_shape as model

v, two, run, hard, m = prior.v, prior.two, prior.run, prior.hard, prior.m
OUT = prior.OUT.parent/"opendd_reserve_shape_v1"
CLOCKS = ("gain", "easy", "incumbent")
ARMS = ("floor", "neural_frozen", "neural_matched", "ols4_matched", "hindcast_matched")
CONTRASTS = (("ols4_matched", "neural_matched"), ("hindcast_matched", "neural_matched"),
             ("hindcast_matched", "ols4_matched"), ("neural_matched", "neural_frozen"))


def register():
    key, outer, old, source, _, clock, _ = prior.register()
    files = [prior.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "absolute")]
    a, h, z = map(run.read, files)
    if (a["registration"] != key or h["registration"] != key or a["heads"] != run.sha(files[1])
            or z["analysis"] != run.sha(files[0]) or any(x["confirmation_access"] or not x["original_2pct"] for x in (a, z))):
        raise ValueError("Require frozen causal reserve outputs")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(f) for f in files], native_steps=[8, 12], seeds=two.study.prior.SEEDS,
        clocks=CLOCKS, arms=ARMS, candidate_bank=model.ANALYTIC, causal_hindcast_prefixes=[4, 5, 6],
        primary="matched OLS4 versus neural at the frozen gain-clock rows and common peak radius",
        radius="minimum native peak over frozen neural, OLS4 and selected analytic; no amplification",
        incumbent_secondary="all original edits, full neural radius before matching; no reserve",
        zero_radius="retain the original neural on all matched arms, report coverage",
        no_new_training=True, no_threshold_search=True, original_2pct=True, confirmation_access=False,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(p), outer, old, source, clock, a, h["heads"]


@torch.inference_mode()
def evaluate(d, bank, path, starts, release, clock, net, cut, ref):
    cv, ols = bank[:2]
    q, rec = d["query"], d["rec"]
    p = cv+prior.model.AMPLITUDE*(path-cv)
    x, scale = prior.feature(d, cv, p)
    gain = net(torch.from_numpy(x)).numpy()[:, 0]*scale
    easy = release(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()[:, 0]
    legal = (p != cv).any((1, 2))
    selected, base = {k: [] for k in CLOCKS[:2]}, cv.copy()
    for r, seq in study.packets(d, cv, ols, path, starts, release):
        plan, _ = m.stream(iter(seq), lambda x: clock(torch.from_numpy(x)).numpy(), "adaptive_queue")
        packets = []
        for j, (used, _, _) in zip(np.unique(q[rec == r]), plan, strict=True):
            base[used] = path[used]
            at = np.setdiff1d(np.flatnonzero((q == j) & legal), used, assume_unique=True)
            packets.append((at, easy[at], gain[at], easy[at]))
        for k in selected:
            match = prior.model.first(iter(packets), k)
            if match is not None:
                selected[k].append(match[1])

    c, o, a, family = model.analytic(d["h"])
    np.testing.assert_allclose(c, cv, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(o, ols, atol=1e-12, rtol=0.)
    y, valid = d["y"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    result = {}
    selected["incumbent"] = np.flatnonzero((base != cv).any((1, 2)))
    for k, rows in selected.items():
        at = np.asarray(rows, dtype=int)
        floor, frozen = (cv, base) if k == "incumbent" else (base, p)
        if k != "incumbent" and (len(np.unique(rec[at])) != len(at) or not (base[at] == cv[at]).all()):
            raise ValueError("A frozen slot overlaps the incumbent")
        common, legal, radius = model.match(cv[at], frozen[at], ols[at], a[at])
        for z in common:
            np.testing.assert_allclose(model.peak(z[legal]-cv[at][legal]), radius[legal], atol=1e-10, rtol=1e-8)
        raw = {"floor": floor, **{name: floor.copy() for name in ARMS[1:]}}
        raw["neural_frozen"][at] = frozen[at]
        for name, z in zip(ARMS[2:], common, strict=True):
            raw[name][at] = z
        for name in ARMS[1:]:
            np.testing.assert_array_equal(np.flatnonzero((raw[name] != floor).any((1, 2))), np.sort(at))
        stats = {name: v.parent.score(d, cv, ols, path2, cut) for name, path2 in raw.items()}
        checks = (("neural_frozen", "base"),) if k == "incumbent" else (("floor", "base"), ("neural_frozen", k))
        for name, before in checks:
            if any(stats[name][f] != ref["metrics"][before][f] for f in v.prior.FIELDS):
                raise ValueError("Did not recover the frozen reserve clock")
        effects = {f"{a}_vs_{b}": hard.paired_pp(raw[b], raw[a], y, valid, mass) for a, b in CONTRASTS}
        effects.update({f"{a}_vs_cv": hard.paired_pp(cv, raw[a], y, valid, mass) for a in ARMS})
        effect = study.c.relative_effect(d, raw["neural_matched"], raw["ols4_matched"])
        endpoint = valid[at, -1]
        errors = {name: dict(fde_mass=float(np.linalg.norm(path2[at, -1]-y[at, -1], axis=-1)[endpoint].sum()),
                             ade_mass=float(np.linalg.norm(path2[at]-y[at], axis=-1)[valid[at]].sum()))
                  for name, path2 in raw.items()}
        result[k] = dict(metrics=stats, effects=effects, relative_ols_vs_neural=effect,
            coverage=dict(slots=len(at), radius_matched=int(legal.sum()), endpoint_rows=int(endpoint.sum()),
                          observed_steps=int(valid[at].sum()), frozen_radius=float(model.peak(frozen[at]-cv[at]).sum()),
                          common_radius=float(radius[legal].sum()+model.peak(frozen[at][~legal]-cv[at][~legal]).sum())),
            families=Counter(model.ANALYTIC[i] for i in family[at][legal]), error_mass=errors)
    return result


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    key, outer, old, source, clock, a, heads = register()
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    results = {}
    for s, (d, _, starts) in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z[0]["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        results[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            release = two.fit.load(outer[f"{s}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            p = next(p for _, seq in study.packets(d, *bank[:2], path, starts, release) for p in seq)
            n = p.features.shape[1]+m.QUEUE_WIDTH
            inc = two.fit.load(clock[f"{s}_{seed}_full"], lambda: v.model.network(n), study.STEPS)
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            net = two.fit.load(heads[f"{s}_{seed}"], lambda: prior.model.network(n), prior.STEPS)
            ref = next(z for z in a["by_site"][s] if z["seed"] == seed)
            results[s].append(dict(seed=seed, **evaluate(d, bank, path, starts, release, inc, net, cut, ref)))
        print(s, "fixed-slot shape readout complete", flush=True)
    flat = [r for rows in results.values() for r in rows]

    def ci(fn):
        cells = [[fn(r) for r in rows] for rows in results.values()]
        x = np.array([np.mean(v) for v in cells if all(y is not None for y in v)])
        if not len(x):
            return None
        draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()), sites=len(x))

    summary = {}
    for k in CLOCKS:
        summary[k] = dict(metrics={name: {f: sum(r[k]["metrics"][name][f] for r in flat) for f in v.prior.FIELDS}
                                   for name in ARMS},
            effects={j: ci(lambda r: r[k]["effects"][j]) for j in flat[0][k]["effects"]},
            relative=ci(lambda r: r[k]["relative_ols_vs_neural"]),
            coverage={f: sum(r[k]["coverage"][f] for r in flat) for f in flat[0][k]["coverage"]},
            families=sum((r[k]["families"] for r in flat), Counter()),
            errors={name: {f: sum(r[k]["error_mass"][name][f] for r in flat) for f in ("fde_mass", "ade_mass")}
                    for name in ARMS})
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, by_site=results, summary=summary,
        original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
