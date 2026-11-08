"""Test direct within-count ranking at the exact frozen OpenDD slate clock."""

import argparse
from collections import Counter
import fcntl
from itertools import combinations
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from src.world_model import m3w_opendd_slate_rank as model

v, run, two, hard, m = study.v, study.run, study.two, study.hard, study.m
OUT, PRIVATE = study.OUT.parent/"opendd_slate_rank_v1", study.PRIVATE.parent/"opendd_slate_rank_v1"
ARMS = ("fixed", "old", "scalar", "paired", "guarded", "oracle_local")
STEPS = 500


def register():
    key, outer, old, source, _, producers, releases, _ = study.register()
    files = [study.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "clock_registration", "clock_capacity")]
    a, heads, cr, cap = map(run.read, files)
    if (a["registration"] != key or heads["registration"] != key or a["heads"] != run.sha(files[1])
            or cap["registration"] != run.sha(files[2]) or cr["parent"]["sha256"] != run.sha(files[0])
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (a, cr, cap))):
        raise ValueError("Require verified frozen slate and realized clock")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(f) for f in files], seeds=two.study.prior.SEEDS, steps=STEPS, workers=0, threads=4,
        native_steps=[8, 12], arms=ARMS, primary="paired vs fixed on exact frozen held clock",
        all_arms_match_query_counts=True, gain="signed interval-separated pair advantage; ambiguous gain maps to zero",
        risks=["all_excess", "easy_excess", "easy_signed"],
        guard="strict predicted gain and no predicted excess increases relative to same-count fixed rank",
        donor_examples="all pairs at counts reachable from current and two prior releases, with both donor and outer excluded from forecaster and release head",
        oracle="local lower utility against fixed; no joint safety implication",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"),
        original_2pct=True, no_threshold_search=True, confirmation_access=False)))
    return run.sha(p), outer, old, source, producers, releases, heads["heads"], a


def row_targets(d, cv, path, cut):
    known, lower = v.paired(cv, path, d["y"], d["valid"])
    g = known.mean(1)
    debit = (known-lower).mean(1)
    moments = two.moments(cv, path, d["y"], d["valid"], cut)
    r = np.column_stack([moments[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    proxy = lower.mean(1)-r.clip(min=0.) @ [2., 4., 1.]
    return g, debit, r, proxy


def training(d, cv, ols, path, starts, head, cut):
    targets = row_targets(d, cv, path, cut)
    x, z = [], []
    for _, seq in study.packets(d, cv, ols, path, starts, head):
        recent = [0, 0]
        for p in seq:
            limit = min(len(p.fixed)-1, sum(recent)+p.release)
            for pair, labels in model.examples(p, limit, *targets):
                x.append(pair)
                z.append(labels)
            recent = [recent[-1], p.release]
    if not x:
        return None
    return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(np.asarray(z, dtype=np.float32))


@torch.inference_mode()
def evaluate(d, bank, path, starts, release, old, scalar, rank, cut, ref):
    cv, ols = bank[:2]
    y, valid, q = (d[k] for k in ("y", "valid", "query"))
    pred = {k: cv.copy() for k in ARMS}
    targets = row_targets(d, cv, path, cut[0])
    comparisons, changed = Counter(), Counter()
    for _, seq in study.packets(d, cv, ols, path, starts, release):
        plan, _ = m.stream(iter(seq), lambda x: old(torch.from_numpy(x)).numpy(), "adaptive_queue")
        for p, (chosen, fixed, _) in zip(seq, plan, strict=True):
            if not len(chosen):
                continue
            ids = np.flatnonzero(p.size == len(chosen))
            f = int(np.flatnonzero(ids == p.fixed[len(chosen)])[0])
            x = torch.from_numpy(p.features[ids])
            scores, baseline = rank(x).numpy(), scalar(x).numpy()
            decisions = dict(scalar=model.decision(baseline, f), paired=model.decision(scores, f),
                             guarded=model.decision(scores, f, True))
            alternatives = [p.rows[i] for i in ids]
            # All causal decisions are fixed before reading query outcomes for diagnostics.
            vs_fixed = [model.matched_advantage(a, fixed, *targets[:2]) for a in alternatives]
            value = np.array([center-radius for center, radius in vs_fixed])
            decisions["oracle_local"] = int(value.argmax()) if value.max() > 0. else f
            comparisons.update(active=1, choice=int(len(ids) > 1), positive_opportunity=int(value.max() > 0.))
            for i, j in combinations(range(len(ids)), 2):
                g, debit = model.matched_advantage(alternatives[i], alternatives[j], *targets[:2])
                if abs(g) > debit:
                    comparisons["resolved_pairs"] += 1
                    comparisons["paired_correct"] += int(np.sign(scores[i, 0]-scores[j, 0]) == np.sign(g))
                    comparisons["scalar_correct"] += int(np.sign(baseline[i, 0]-baseline[j, 0]) == np.sign(g))
            pred["old"][chosen], pred["fixed"][fixed] = path[chosen], path[fixed]
            for k, j in decisions.items():
                at = alternatives[j]
                pred[k][at] = path[at]
                changed[k] += len(set(at)-set(fixed))
    counts = lambda p: np.bincount(q[(p != cv).any((1, 2))], minlength=q.max()+1)
    edits = counts(pred["old"])
    for p in pred.values():
        np.testing.assert_array_equal(counts(p), edits)
    stats = {k: v.parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
    for name, prev in (("old", "adaptive_queue"), ("fixed", "same_clock")):
        for f in v.prior.FIELDS:
            if stats[name][f] != ref["metrics"][prev][f]:
                raise ValueError("Frozen controls changed")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(a, "fixed") for a in ARMS[1:]] + [(a, "cv") for a in ("fixed", "old", "scalar", "paired", "guarded")]
    pairs += [(a, "scalar") for a in ("paired", "guarded")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass) for a, b in pairs}
    easy = {}
    for k, p in pred.items():
        risk = two.moments(cv, p, y, valid, cut[0])
        denom = risk["ref_easy"].sum()
        easy[k] = float(100.*risk["easy_signed"].sum()/denom) if denom > 0. else None
    return dict(metrics=stats, effects=effects, easy_net_percent=easy, comparisons=comparisons, changed=changed)


def execute(key, outer, old, source, producers, releases, old_heads, ref, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(z[0]) for s, z in sites.items()}
    maps = {(tuple(z["excluded"]), z["target"], z["seed"]): z for z in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in use]), [.25, .75])
    width = hard.model.features(next(iter(sites.values()))[0]["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0
    for held, (d, _, starts) in sites.items():
        fit = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            train = []
            for s in fit:
                pair = tuple(sorted((held, s)))
                h = releases["release_"+"_".join(pair)+f"_{seed}"]
                if set(h["identity"]["fit"]) & set(pair):
                    raise ValueError("Release head exposed to the donor or held site")
                p = np.load(run.checked(maps[pair, s, seed]["prediction"]), allow_pickle=False)
                rel = two.fit.load(h, lambda: hard.model.network(width), hard.STEPS)
                t = training(sites[s][0], *banks[s][:2], p, sites[s][2], rel, cuts([u for u in fit if u != s])[0])
                if t is not None:
                    train.append(t)
            if not train:
                raise ValueError("No causal fitting pairs")
            n = train[0][0].shape[-1]
            nets = []
            for scalar in (True, False):
                arm = "scalar" if scalar else "paired"
                name = f"{held}_{seed}_{arm}"
                ident = dict(registration=key, held=held, fit=fit, seed=seed, arm=arm)
                net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(n, scalar), STEPS,
                    two.batcher(train), lambda net, x, y: model.loss(net, x, y, scalar), replay, beat)
                updates += u
                nets.append(net)
            row = next(z for z in old["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                p = z["comoving_ego"]
            rel = two.fit.load(outer[f"{held}_{seed}"], lambda: hard.model.network(width), hard.STEPS)
            clock = two.fit.load(old_heads[f"{held}_{seed}_full"], lambda: v.model.network(n+m.QUEUE_WIDTH), study.STEPS)
            a = next(z for z in ref["by_site"][held] if z["seed"] == seed)
            r = evaluate(d, banks[held], p, starts, rel, clock, *nets, cuts(fit), a)
            result[held].append(dict(seed=seed, donor_pairs=sum(len(x) for x, _ in train), **r))
            beat(phase="site", site=held, seed=seed, updates=updates)
    flat = [r for rows in result.values() for r in rows]

    def ci(fn):
        values = [[fn(r) for r in group] for group in result.values()]
        x = np.array([np.mean(z) for z in values if all(v is not None for v in z)])
        if not len(x):
            return None
        draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()), sites=len(x))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(lambda r: r["effects"][k]) for k in flat[0]["effects"]},
        easy_net_percent={k: ci(lambda r: r["easy_net_percent"][k]) for k in ARMS},
        comparisons=sum((r["comparisons"] for r in flat), Counter()), changed=sum((r["changed"] for r in flat), Counter()))
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**p):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **p)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", z)
            if p.get("phase") != "fit" or p["step"] == STEPS:
                print(z, flush=True)
        two.fit.PRIVATE = PRIVATE
        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
