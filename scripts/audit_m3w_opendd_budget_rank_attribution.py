"""Decompose frozen reranking by recipient changes versus shape-only changes."""

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
from scripts import run_m3w_opendd_budget_rank as rank
from src.evaluation.m3w_selection_forecast_readout import policy

v, run, two, hard, prior, model = rank.v, rank.run, rank.two, rank.hard, rank.prior, rank.model
OUT = rank.OUT
ARMS = ("original", "rank", "shape_only", "recipient_changes")


@torch.inference_mode()
def evaluate(d, bank, path, release, gain, net, analytic, cut, ref):
    cv, ols = bank[:2]
    y, valid, rec, query = (d[k] for k in ("y", "valid", "rec", "query"))
    pred = {k: cv.copy() for k in ARMS}
    counts = Counter()
    for at, frozen, groups in rank.packets(d, cv, ols, path, release, gain, analytic):
        for p in pred.values():
            p[at] = frozen
        for rows, plans, features, paths, unit in groups:
            z = net(torch.from_numpy(features)).numpy()
            pick = model.choose(z)
            moved = any(j != a for (_, j, _), (_, a, _) in zip(plans[pick], plans[0], strict=True))
            pred["rank"][rows] = paths[pick]
            if pick != 0:
                k = "recipient_changes" if moved else "shape_only"
                pred[k][rows] = paths[pick]
                truth = rank.targets(d, cv, ols, at, frozen, rows, paths, unit, cut[0])[pick]
                counts[k+"_queries"] += 1
                counts[k+"_better_and_feasible"] += int(truth[0] > truth[1] and truth[5] > .5)
                counts[k+"_worse"] += int(truth[0] < -truth[1])
    stats = {k: v.parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
    for k in ("original", "rank"):
        if any(stats[k][f] != ref["metrics"][k][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen policy did not replay")
    qcount = lambda p: np.bincount(query[(p != cv).any((1, 2))], minlength=query.max()+1)
    for k in ARMS:
        np.testing.assert_array_equal(qcount(pred[k]), qcount(pred["original"]))
    risks = {k: policy(p, cv, cv, y, valid, rec, easy_cut=cut[0], hard_cut=cut[1])["recordings"]
             for k, p in pred.items()}
    for k in ARMS[1:]:
        for r, a in risks["original"].items():
            b = risks[k][r]
            before, after = (z["completion_easy_risk_upper"] is not None for z in (a, b))
            counts[k+"_lost_reference"] += int(before and not after)
            counts[k+"_new_reference"] += int(not before and after)
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    effects = {k+"_vs_original": hard.paired_pp(pred["original"], pred[k], y, valid, mass) for k in ARMS[1:]}
    return dict(metrics=stats, effects=effects, diagnostics=counts)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, outer, old, source, gains, *_ = rank.register()
    p, h = (OUT/f"{k}.json" for k in ("analysis", "head_freeze"))
    a, b = map(run.read, (p, h))
    if (a["registration"] != key or a["heads"] != run.sha(h) or b["registration"] != key
            or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require verified matched-budget ranker")
    f = OUT/"attribution_registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), parents=[run.artifact(p), run.artifact(h)],
        original_2pct=True, confirmation_access=False, post_result_development_diagnostic=True,
        no_new_training=True, decomposition="apply frozen rank swaps that keep recipients or change at least one",
        exact_query_counts=True, bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    result = {}
    for s, d in sites.items():
        bank = v.parent.prior.analytic(d)
        cut = np.quantile(np.concatenate([z["cut_samples"] for t, z in sites.items() if t != s]), [.25, .75])
        analytic = rank.shape.analytic(d["h"])[2]
        width = hard.model.features(d["h"][:1]).shape[1]
        result[s] = []
        for row in old["by_site"][s]:
            seed = row["seed"]
            name = f"{s}_{seed}"
            release = two.fit.load(outer[name], lambda: hard.model.network(width), hard.STEPS)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            gain = two.fit.load(gains[name], lambda: prior.model.network(n), prior.STEPS)
            memo = b["heads"][name+"_rank"]
            if memo["identity"]["held"] != s or memo["identity"]["arm"] != "rank":
                raise ValueError("Wrong frozen ranker")
            packets = rank.packets(d, *bank[:2], path, release, gain, analytic)
            first = next(g[2].shape[1] for _, _, groups in packets for g in groups)
            net = two.fit.load(memo, lambda: model.network(first), rank.STEPS)
            ref = next(z for z in a["by_site"][s] if z["seed"] == seed)
            result[s].append(dict(seed=seed, **evaluate(d, bank, path, release, gain, net, analytic, cut, ref)))
        print(s, "rank swap attribution complete", flush=True)
    flat = [r for group in result.values() for r in group]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in group]) for group in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]}, diagnostics=sum((r["diagnostics"] for r in flat), Counter()))
    run.once(OUT/"attribution_analysis.json", run.stable(dict(registration=run.sha(f), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))
    print(run.stable(summary), flush=True)


if __name__ == "__main__":
    main()
