"""Cross-fit candidate-conditioned support heads at the frozen second slot."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch
from torch.nn.functional import softplus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_support_stop as stop
from src.world_model import m3w_opendd_support_admission as model

v, run, two, hard, prior, base = stop.v, stop.run, stop.two, stop.hard, stop.prior, stop.base
OUT = stop.OUT.parent/"opendd_support_admission_v1"
PRIVATE = stop.PRIVATE.parent/OUT.name
LEARNED = ("local_gain", "local_support", "restricted_support", "guarded_support")
ARMS = ("gain", "easy", "matched_gain", *LEARNED)
STEPS = 500


def register():
    k, outer, old, source, gain, prev, producers, releases, _ = stop.register()
    parents = [stop.OUT/f"{a}.json" for a in ("analysis", "menu_registration", "menu_capacity")]
    a, m, c = map(run.read, parents)
    if (a["registration"] != k or c["registration"] != run.sha(parents[1])
            or m["parent"]["sha256"] != run.sha(parents[0])
            or not c["outcome_aware_only"] or
            any(z["confirmation_access"] or not z["original_2pct"] for z in (a, m, c))):
        raise ValueError("Require bound stopping and menu evidence")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(p) for p in parents],
        native_steps=[8, 12], seeds=two.study.prior.SEEDS, threads=4, workers=0, steps=STEPS,
        matching="first gain edit and second query, count, amplitude fixed; only recipient changes",
        training="all donor queries with an easy anchor, double-excluded forecast and easy head; no donor first-slot exclusion",
        features="current causal candidate features, differences from current easy anchor, legal menu size",
        targets="normalized conservative CV gain; local easy and risk-feasible positive event; normalized positive excess",
        primary="guarded support vs frozen easy at same query; needs positive lower utility and no reduction in joint counts",
        controls=ARMS, hard_rule_unchanged=True, no_threshold_search=True,
        original_2pct=True, confirmation_access=False,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), outer, old, source, gain, prev, producers, releases, c


def targets(d, cv, path, scale, cut):
    y, valid = d["y"], d["valid"]
    mom = two.moments(cv, path, y, valid, cut)
    lower = v.paired(cv, path, y, valid)[1].mean(1)
    risk = np.column_stack([mom[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    hit = mom["easy"] & (lower > 0.) & (risk[:, :2] <= 0.).all(1) & (risk[:, 2] <= .02*mom["ref_easy"])
    harm = risk.clip(min=0.) @ [2., 4., 1.]
    return np.c_[(lower/scale).clip(-8., 8.), hit, (harm/scale).clip(0., 8.)].astype(np.float32), mom["easy"]


@torch.inference_mode()
def training(d, cv, path, release, cut):
    p, x, scale, easy, blocks = stop.inputs(d, cv, path, release)
    target, _ = targets(d, cv, p, scale, cut)
    features, labels = [], []
    for seq in blocks.values():
        for rows in seq:
            rows, anchor = model.menu(rows, easy)
            if anchor is not None:
                features.append(model.features(x, easy, rows, anchor))
                labels.append(target[rows])
    if labels:
        return tuple(torch.from_numpy(np.concatenate(z)) for z in (features, labels))
    return None


@torch.inference_mode()
def evaluate(d, bank, path, release, gain_head, net, cut, ref, ceiling):
    cv, ols = bank[:2]
    p, x, scale, easy, blocks = stop.inputs(d, cv, path, release)
    large = cv+base.model.GAIN_SCALE*(path-cv)
    f, size = prior.feature(d, cv, large)
    gain = gain_head(torch.from_numpy(f)).numpy()[:, 0]*size
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    pred = {k: cv.copy() for k in ARMS}
    choices = []
    for seq in blocks.values():
        first, second, matched = base.model.choose((rows, gain[rows], easy[rows]) for rows in seq)
        if first is not None:
            for out in pred.values():
                out[first] = large[first]
        if second is None:
            continue
        rows = next(rows for rows in seq if len(rows) and q[rows[0]] == q[second])
        rows, anchor = model.menu(rows, easy, first)
        if anchor != second or matched not in rows:
            raise ValueError("Frozen query menu changed")
        z = net(torch.from_numpy(model.features(x, easy, rows, anchor)))
        z[:, 2] = softplus(z[:, 2])
        chosen = dict(easy=anchor, matched_gain=matched, **model.choose(rows, anchor, easy, z.numpy()))
        for k, j in chosen.items():
            if j not in rows or rec[j] != rec[second] or q[j] != q[second]:
                raise ValueError("Admission cannot move or duplicate an edit")
            pred[k][j] = p[j]
        choices.append(chosen)
    stats = {k: v.parent.score(d, cv, ols, z, cut) for k, z in pred.items()}
    for k, r in (("gain", "gain"), ("easy", "two_roles"), ("matched_gain", "two_gain_matched")):
        if any(stats[k][f] != ref["metrics"][r][f] for f in v.prior.FIELDS):
            raise ValueError("Failed to recover exact frozen references")
    for k in ARMS[1:]:
        if (stats[k]["edits"] != stats["gain"]["edits"]+len(choices)
                or stats[k]["dual_positive"] > ceiling["fixed"]["all_legal"]):
            raise ValueError("Second slot count or oracle capacity changed")
    labels, truth = targets(d, cv, p, scale, cut[0])
    detail = {k: Counter() for k in ARMS[1:]}
    for z in choices:
        for k, j in z.items():
            detail[k].update(support=1, different=int(j != z["easy"]),
                             outside_easy=int(easy[j] < base.model.EASY_THRESHOLD),
                             observed_easy=int(truth[j]), observed_local_support=int(labels[j, 1]))
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+[(k, "easy") for k in LEARNED]
    pairs += [("local_support", "restricted_support"), ("guarded_support", "matched_gain")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass)
               for a, b in pairs}
    return dict(metrics=stats, effects=effects, detail=detail)


def execute(key, outer, old, source, gain_heads, prev, producers, releases, capacity, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(z[0]) for s, z in sites.items()}
    receipts = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))[0]["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0
    for held, (d, _, _) in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit = []
            for s in use:
                pair = tuple(sorted((s, held)))
                h = releases["release_"+"_".join(pair)+f"_{seed}"]
                if set(h["identity"]["fit"]) & set(pair):
                    raise ValueError("Donor easy head used excluded labels")
                release = two.fit.load(h, lambda: hard.model.network(hard_width), hard.STEPS)
                path = np.load(run.checked(receipts[pair, s, seed]["prediction"]), allow_pickle=False)
                z = training(sites[s][0], banks[s][0], path, release, cuts([j for j in use if j != s])[0])
                if z is not None:
                    fit.append(z)
            name = f"{held}_{seed}"
            width = fit[0][0].shape[1]
            identity = dict(registration=key, held=held, fit=use, seed=seed)
            net, freeze[name], u = two.fit.fit(identity, name, seed, lambda: model.network(width), STEPS,
                                               two.batcher(fit), model.loss, replay, beat)
            updates += u
            release = two.fit.load(outer[name], lambda: hard.model.network(hard_width), hard.STEPS)
            lookup = lambda x: next(z for z in x["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            bank = banks[held]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[0][:1], path[:1])[0].shape[1]
            gain_head = two.fit.load(gain_heads[name], lambda: prior.model.network(n), prior.STEPS)
            cell = evaluate(d, bank, path, release, gain_head, net, cuts(use), lookup(prev), lookup(capacity)["counts"])
            rows = sum(len(x) for x, _ in fit)
            positive = sum(int(y[:, 1].sum()) for _, y in fit)
            result[held].append(dict(seed=seed, donor_rows=rows, donor_support_positive=positive, **cell))
            beat(phase="site", site=held, seed=seed, updates=updates, rows=rows, positive=positive)
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]},
        detail={k: sum((r["detail"][k] for r in flat), Counter()) for k in ARMS[1:]})
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
