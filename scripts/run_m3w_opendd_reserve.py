"""Prospective one-slot support intervention on the exposed OpenDD sites."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import expit

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import audit_m3w_opendd_one_extra_edit as prior
from src.world_model import m3w_opendd_reserve as model

v, run, two, hard, m = prior.v, prior.run, prior.two, prior.hard, prior.m
OUT, PRIVATE = prior.OUT.parent/"opendd_reserve_v1", prior.teacher.PRIVATE.parent/"opendd_reserve_v1"
ARMS = ("base", "easy", "gain", "joint", "easy_same_query", "gain_same_query")
STEPS = 500


def register():
    key, outer, old, source, producers, _, clock, _, _ = prior.teacher.register()
    files = [prior.OUT/f"{k}.json" for k in ("analysis", "support_registration", "support_capacity")]
    a, r, c = map(run.read, files)
    if (a["registration"] != key or c["registration"] != run.sha(files[1])
            or r["parents"][0]["sha256"] != run.sha(files[0]) or r["changes_main_budget"]
            or not c["outcome_aware_only"] or any(z["confirmation_access"] or not z["original_2pct"] for z in (a, r, c))):
        raise ValueError("Require the prior outcome-aware support evidence")
    p = OUT/"registration.json"
    run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(f) for f in files], steps=STEPS, seeds=two.study.prior.SEEDS,
        threads=4, workers=0, native_steps=[8, 12], arms=ARMS, amplitude=model.AMPLITUDE,
        joint_probability=model.JOINT_THRESHOLD, easy_probability=model.EASY_THRESHOLD,
        training="cross-fitted candidate rows; predict lower CV gain, easy support and local easy/risk/positive event",
        slot="one extra available from the start of every recording; original edits not replaced; no terminal information",
        budget="separate development arm, primary fixed-budget results unchanged",
        matching="same_query arms use exactly the joint arm's recording, time, extra count and amplitude",
        no_threshold_search=True, original_2pct=True, confirmation_access=False,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(p), outer, old, source, producers, clock, c


def feature(d, cv, p):
    x, scale = study.c.model.features(d["h"], cv, p)
    q, origin = d["query"], d["h"][:, -1]
    context = np.zeros((len(q), 2), dtype=np.float32)
    for j in np.unique(q):
        ix = np.flatnonzero(q == j)
        context[ix, 0] = np.log1p(len(ix))
        if len(ix) > 1:
            dist = np.linalg.norm(origin[ix, None]-origin[None, ix], axis=-1)
            np.fill_diagonal(dist, np.inf)
            context[ix, 1] = np.log1p(dist.min(1)/scale[ix]).clip(0., 16.)
    return np.c_[x, context], scale


def training(d, cv, path, cut):
    p = cv+model.AMPLITUDE*(path-cv)
    x, scale = feature(d, cv, p)
    mom = two.moments(cv, p, d["y"], d["valid"], cut)
    lower = v.paired(cv, p, d["y"], d["valid"])[1].mean(1)
    hit = mom["easy"] & (mom["all_excess"] <= 0.) & (mom["easy_excess"] <= 0.) & (lower > 0.)
    legal = (p != cv).any((1, 2))
    target = np.c_[(lower/scale).clip(-8., 8.), hit, mom["easy"]].astype(np.float32)
    if not legal.any():
        return None
    return torch.from_numpy(x[legal]), torch.from_numpy(target[legal])


@torch.inference_mode()
def evaluate(d, bank, path, starts, release, clock, net, cut, prior_cell, ceiling):
    cv, ols = bank[:2]
    p = cv+model.AMPLITUDE*(path-cv)
    y, valid, q, rec = (d[k] for k in ("y", "valid", "query", "rec"))
    x, scale = feature(d, cv, p)
    z = net(torch.from_numpy(x)).numpy()
    gain, joint = z[:, 0]*scale, expit(z[:, 1])
    easy = release(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()[:, 0]
    legal = (p != cv).any((1, 2))
    plans = {k: {} for k in ARMS[1:]}
    base, frozen = cv.copy(), {}
    for r, seq in study.packets(d, cv, ols, path, starts, release):
        old, _ = m.stream(iter(seq), lambda x: clock(torch.from_numpy(x)).numpy(), "adaptive_queue")
        ids = np.unique(q[rec == r])
        if len(ids) != len(old):
            raise ValueError("The packet and original clock disagree")
        frozen[r] = seq, [len(z[0]) for z in old]
        packets = []
        for j, (used, _, _) in zip(ids, old, strict=True):
            base[used] = path[used]
            at = np.setdiff1d(np.flatnonzero((q == j) & legal), used, assume_unique=True)
            packets.append((at, easy[at], gain[at], joint[at]))
        for arm in ("easy", "gain", "joint"):
            plan = model.first(iter(packets), arm)
            if plan is not None:
                j, row = plan
                plans[arm][r] = row
                if arm == "joint":
                    for k in ("easy", "gain"):
                        plans[f"{k}_same_query"][r] = model.pick(*packets[j], k, force=True)
    for k in ("easy_same_query", "gain_same_query"):
        a, b = plans["joint"], plans[k]
        if a.keys() != b.keys() or any(q[a[r]] != q[b[r]] for r in a):
            raise ValueError("Same-query reserve comparison changed coverage or time")

    pred, details, stats = {"base": base}, {}, {}
    for k, chosen in plans.items():
        a = np.asarray(list(chosen.values()), dtype=int)
        if len(a) and (not (base[a] == cv[a]).all() or not legal[a].all() or len(np.unique(rec[a])) != len(a)):
            raise ValueError("A reserve edited an incumbent or reused its token")
        pred[k] = base.copy()
        pred[k][a] = p[a]
        mom = two.moments(cv, p, y, valid, cut[0])
        lower = v.paired(cv, p, y, valid)[1].sum(1)
        details[k] = dict(extras=len(a), easy=int(mom["easy"][a].sum()),
            harmful_lower=int((lower[a] < 0.).sum()), unknown=int((~mom["known"][a]).sum()))
    for k, out in pred.items():
        stats[k] = v.parent.score(d, cv, ols, out, cut)
    for f in v.prior.FIELDS:
        if stats["base"][f] != prior_cell["metrics"]["adaptive_queue"][f]:
            raise ValueError("The underlying frozen policy changed")
    for k in ARMS[1:]:
        if stats[k]["edits"]-stats["base"]["edits"] != details[k]["extras"]:
            raise ValueError("The measured edit count differs from the prospective reserve plan")

    # Cohorts are assigned after choices, never supplied to the stream.
    support = two.moments(cv, cv, y, valid, cut[0])["easy"]
    cohorts = {k: {a: Counter() for a in ARMS[1:]} for k in prior.GROUPS}
    for r, (seq, counts) in frozen.items():
        if not sum(counts):
            group = "no_credits"
        elif not any(support[p.rows[i]].any() for p, n in zip(seq, counts, strict=True) if n
                     for i in np.flatnonzero(p.size == n)):
            group = "no_reachable_easy"
        else:
            continue
        at = rec == r
        sub = dict(y=y[at], valid=valid[at], rec=rec[at])
        for k in ARMS[1:]:
            a = v.parent.score(sub, cv[at], ols[at], pred[k][at], cut)
            cohorts[group][k].update(total=1, extra=int(r in plans[k]), joint=a["dual_positive"])
    for group in prior.GROUPS:
        for k in ARMS[1:]:
            a, b = cohorts[group][k], ceiling[group]
            if a["total"] != b.get("total", 0) or a["joint"] > b.get("qualifies_at_0.25", 0):
                raise ValueError("Causal support result exceeds the matching outcome-aware ceiling")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "base") for k in ARMS[1:]]+[(k, "cv") for k in ("base", "joint")]
    pairs += [("joint", k) for k in ("easy_same_query", "gain_same_query")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass) for a, b in pairs}
    return dict(metrics=stats, effects=effects, reserve=details, cohorts=cohorts)


def execute(key, outer, old, source, producers, clock, cap, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    bank = {s: v.parent.prior.analytic(z[0]) for s, z in sites.items()}
    refs = {(tuple(p["excluded"]), p["target"], p["seed"]): p for p in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))[0]["h"][:1]).shape[1]
    source_readout = run.read(study.OUT/"analysis.json")["by_site"]
    result, freeze, updates = {}, {}, 0
    for held, (d, _, starts) in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit = []
            for s in use:
                pair = tuple(sorted((s, held)))
                path = np.load(run.checked(refs[pair, s, seed]["prediction"]), allow_pickle=False)
                b = training(sites[s][0], bank[s][0], path, cuts([u for u in use if u != s])[0])
                if b is not None:
                    fit.append(b)
            width = fit[0][0].shape[1]
            name = f"{held}_{seed}"
            ident = dict(registration=key, held=held, fit=use, seed=seed)
            net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(width), STEPS,
                                               two.batcher(fit), model.loss, replay, beat)
            updates += u
            rel = two.fit.load(outer[f"{held}_{seed}"], lambda: hard.model.network(hard_width), hard.STEPS)
            row = next(z for z in old["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            first = next(p for _, seq in study.packets(d, *bank[held][:2], path, starts, rel) for p in seq)
            n = first.features.shape[1]+m.QUEUE_WIDTH
            incumbent = two.fit.load(clock[f"{held}_{seed}_full"], lambda: v.model.network(n), study.STEPS)
            previous = next(r for r in source_readout[held] if r["seed"] == seed)
            bound = next(r["groups"] for r in cap["by_site"][held] if r["seed"] == seed)
            cell = evaluate(d, bank[held], path, starts, rel, incumbent, net, cuts(use), previous, bound)
            result[held].append(dict(seed=seed, **cell))
            beat(phase="site", site=held, seed=seed, updates=updates, used=cell["reserve"]["joint"]["extras"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in group]) for group in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]},
        reserve={k: sum((Counter(r["reserve"][k]) for r in flat), Counter()) for k in ARMS[1:]},
        cohorts={g: {k: sum((r["cohorts"][g][k] for r in flat), Counter()) for k in ARMS[1:]} for g in prior.GROUPS})
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
