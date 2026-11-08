"""Ablate incumbent outputs while replaying their causal eligibility trace."""

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
from scripts import run_m3w_opendd_reserve_shape as shape

prior, v, two, run, hard, m = shape.prior, shape.v, shape.two, shape.run, shape.hard, shape.m
OUT = prior.OUT.parent/"opendd_reserve_only_v1"
ARMS = ("cv", "incumbent", "full_gain", "reserve_only", "reserve_free", "reserve_ols_matched", "reserve_neural_matched")


@torch.inference_mode()
def evaluate(d, bank, path, starts, release, clock, net, cut, ref):
    cv, ols = bank[:2]
    p = cv+prior.model.AMPLITUDE*(path-cv)
    x, scale = prior.feature(d, cv, p)
    gain = net(torch.from_numpy(x)).numpy()[:, 0]*scale
    q, rec, y, valid = (d[k] for k in ("query", "rec", "y", "valid"))
    legal = (p != cv).any((1, 2))
    base, chosen, free = cv.copy(), [], []
    for r, seq in study.packets(d, cv, ols, path, starts, release):
        plan, _ = m.stream(iter(seq), lambda z: clock(torch.from_numpy(z)).numpy(), "adaptive_queue")
        picked, picked_free = False, False
        for j, (used, _, _) in zip(np.unique(q[rec == r]), plan, strict=True):
            base[used] = path[used]
            rows = np.flatnonzero((q == j) & legal)
            for at, flag, output in ((np.setdiff1d(rows, used, assume_unique=True), picked, chosen),
                                     (rows, picked_free, free)):
                if not flag and len(at) and gain[at].max() > 0.:
                    output.append(int(at[gain[at].argmax()]))
            picked = bool(chosen and rec[chosen[-1]] == r)
            picked_free = bool(free and rec[free[-1]] == r)
    at, other = np.asarray(chosen, dtype=int), np.asarray(free, dtype=int)
    if (len(np.unique(rec[at])) != len(at) or len(np.unique(rec[other])) != len(other)
            or not (base[at] == cv[at]).all()):
        raise ValueError("A reserve reused its token or conflicted with the incumbent")
    pred = {k: cv.copy() for k in ARMS}
    pred["incumbent"] = base
    pred["full_gain"] = base.copy()
    pred["full_gain"][at] = p[at]
    pred["reserve_only"][at] = p[at]
    pred["reserve_free"][other] = p[other]
    _, _, a, _ = shape.model.analytic(d["h"])
    match, ok, _ = shape.model.match(cv[at], p[at], ols[at], a[at])
    pred["reserve_neural_matched"][at], pred["reserve_ols_matched"][at] = match[:2]
    stats = {k: v.parent.score(d, cv, ols, z, cut) for k, z in pred.items()}
    for k, parent in (("incumbent", "base"), ("full_gain", "gain")):
        if any(stats[k][f] != ref["metrics"][parent][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen gain reserve did not reproduce")
    for k in ("reserve_only", "reserve_ols_matched", "reserve_neural_matched"):
        np.testing.assert_array_equal(np.flatnonzero((pred[k] != cv).any((1, 2))), np.sort(at))
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    contrasts = [(k, "cv") for k in ARMS[1:]]+[("reserve_neural_matched", "reserve_ols_matched")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(pred[b], pred[a], y, valid, mass) for a, b in contrasts}
    expected = ref["effects"]["gain_vs_base"]
    np.testing.assert_allclose(effects["reserve_only_vs_cv"], expected, atol=1e-12, rtol=0.)
    easy = {}
    for k, z in pred.items():
        moments = two.moments(cv, z, y, valid, cut[0])
        den = moments["ref_easy"].sum()
        easy[k] = float(100.*moments["easy_signed"].sum()/den) if den > 0. else None
    return dict(metrics=stats, effects=effects, easy_net_percent=easy,
                support=Counter(matched=int(ok.sum()), reserve=len(at), free=len(other)))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    key, outer, old, source, clock, a, heads = shape.register()
    path = shape.OUT/"analysis.json"
    s = run.read(path)
    if s["registration"] != key or s["confirmation_access"] or not s["original_2pct"]:
        raise ValueError("Require verified matched shape results")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), parent=run.artifact(path),
        source_only_followup=True, no_new_training=True, original_2pct=True, confirmation_access=False,
        primary="reserve only vs CV; original causal shadow path determines the same gain reserve rows",
        free_control="at most one slot without shadow; same positive gain condition and quarter amplitude",
        seeds=two.study.prior.SEEDS, arms=ARMS, native_steps=[8, 12],
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    result = {}
    for s, (d, _, starts) in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z[0]["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            release = two.fit.load(outer[f"{s}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            f = next(p for _, seq in study.packets(d, *bank[:2], path, starts, release) for p in seq)
            n = f.features.shape[1]+m.QUEUE_WIDTH
            inc = two.fit.load(clock[f"{s}_{seed}_full"], lambda: v.model.network(n), study.STEPS)
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            net = two.fit.load(heads[f"{s}_{seed}"], lambda: prior.model.network(n), prior.STEPS)
            ref = next(r for r in a["by_site"][s] if r["seed"] == seed)
            result[s].append(dict(seed=seed, **evaluate(d, bank, path, starts, release, inc, net, cut, ref)))
        print(s, "reserve-only readout complete", flush=True)
    flat = [r for rows in result.values() for r in rows]

    def ci(fn):
        x = np.array([np.mean(z) for z in ([fn(r) for r in rows] for rows in result.values()) if all(y is not None for y in z)])
        if not len(x):
            return None
        draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()), sites=len(x))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(lambda r: r["effects"][k]) for k in flat[0]["effects"]},
        easy_net_percent={k: ci(lambda r: r["easy_net_percent"][k]) for k in ARMS},
        support=sum((r["support"] for r in flat), Counter()))
    run.once(OUT/"analysis.json", run.stable(dict(registration=run.sha(p), by_site=result, summary=summary,
        original_2pct=True, confirmation_access=False, submission_ready=False)))


if __name__ == "__main__":
    main()
