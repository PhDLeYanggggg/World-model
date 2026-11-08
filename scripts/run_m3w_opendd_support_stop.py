"""Cross-fit finite waiting for the second, support-specific OpenDD slot."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_two_reserves as base
from src.world_model import m3w_opendd_support_stop as model

v, run, two, hard, prior = base.v, base.run, base.two, base.hard, base.prior
OUT, PRIVATE = base.OUT.parent/"opendd_support_stop_v1", prior.PRIVATE.parent/"opendd_support_stop_v1"
POLICIES = ("immediate", "wait_one", "myopic", "continue")
ARMS = ("gain", *POLICIES, *(f"{k}_matched" for k in POLICIES[1:]), "bounded_oracle")
STEPS = 400


def register():
    key, outer, old, source, heads, _ = base.register()
    p, t, tr = [base.OUT/f"{k}.json" for k in ("analysis", "timing_capacity", "timing_registration")]
    a, cap, tc = map(run.read, (p, t, tr))
    vkey, _, _, _, _, _, producers = v.register()
    d = v.OUT/"head_freeze.json"
    donor = run.read(d)
    if (a["registration"] != key or cap["registration"] != run.sha(tr) or tc["parent"]["sha256"] != run.sha(p)
            or donor["registration"] != vkey or any(z["confirmation_access"] or not z["original_2pct"] for z in (a, cap, tc))):
        raise ValueError("Require verified two-slot evidence and donor release heads")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(f) for f in (p, t, tr, d)], native_steps=[8, 12],
        seeds=two.study.prior.SEEDS, steps=STEPS, threads=4, workers=0,
        first_slot="unaltered gain-only policy; cannot be displaced by support",
        support="first predicted easy query releases one slot for two more query indices; fixed recipient rank and amplitude",
        train="donor path and easy head both exclude donor and outer site; donor support menu has no first-slot exclusion",
        target="current conservative CV utility with easy-support bonus and risk penalty, minus best future donor value through deadline",
        controls="immediate, wait one, myopic head; same-recording realized-use immediate matches",
        primary="continue vs its same-recording immediate match",
        no_threshold_search=True, original_2pct=True, confirmation_access=False,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), outer, old, source, heads, a, producers, donor["heads"], cap


def options(rows, easy, first):
    out = []
    for ix in rows:
        use = ix[(ix != first) & (easy[ix] >= base.model.EASY_THRESHOLD)]
        out.append(int(use[easy[use].argmax()]) if len(use) else None)
    return out


def describe(x, easy, rows, j, release, i):
    return np.r_[x[i], easy[i], np.log1p(j), j-release, np.log1p(len(rows[j])),
                 np.mean(easy[rows[j]] >= base.model.EASY_THRESHOLD)].astype(np.float32)


def inputs(d, cv, path, release):
    p = cv+base.model.SUPPORT_SCALE*(path-cv)
    x, scale = prior.feature(d, cv, p)
    easy = release(torch.from_numpy(hard.model.features(d["h"]))).softmax(1).numpy()[:, 0]
    legal = (p != cv).any((1, 2))
    q, rec = d["query"], d["rec"]
    blocks = {}
    for r in np.unique(rec):
        at = rec == r
        blocks[r] = [np.flatnonzero(at & (q == j) & legal) for j in np.unique(q[at])]
    return p, x, scale, easy, blocks


@torch.inference_mode()
def training(d, cv, path, release, cut):
    p, x, scale, easy, blocks = inputs(d, cv, path, release)
    y, valid = d["y"], d["valid"]
    mom = two.moments(cv, p, y, valid, cut)
    risks = np.column_stack([mom[k] for k in ("all_excess", "easy_excess", "easy_signed")])
    reward = v.paired(cv, p, y, valid)[1].mean(1)-risks.clip(min=0.) @ [2., 4., 1.]
    reward = (reward/scale+.05*mom["easy"]).clip(-8., 8.)
    a, b = [], []
    for rows in blocks.values():
        candidates = options(rows, easy, None)
        rewards = [reward[i] if i is not None else None for i in candidates]
        ix = list(model.targets(rewards))
        if ix:
            release_t = ix[0][0]
            for t, now, ahead in ix:
                a.append(describe(x, easy, rows, t, release_t, candidates[t]))
                b.append((now, ahead))
    if not a:
        return None
    return torch.from_numpy(np.asarray(a)), torch.from_numpy(np.asarray(b, dtype=np.float32))


@torch.inference_mode()
def evaluate(d, bank, path, release, gain_head, net, cuts, ref, ceiling):
    cv, ols = bank[:2]
    p, x, _, easy, blocks = inputs(d, cv, path, release)
    large = cv+base.model.GAIN_SCALE*(path-cv)
    g, scale = prior.feature(d, cv, large)
    gain = gain_head(torch.from_numpy(g)).numpy()[:, 0]*scale
    y, valid, rec = (d[k] for k in ("y", "valid", "rec"))
    pred = {k: cv.copy() for k in ARMS}
    uses = {k: Counter() for k in POLICIES}
    timing = Counter()
    for r, rows in blocks.items():
        first, old, _ = base.model.choose((ix, gain[ix], easy[ix]) for ix in rows)
        if first is not None:
            for z in pred.values():
                z[first] = large[first]
        if old is None:
            continue
        timing["available"] += 1
        eligible = options(rows, easy, first)
        t0 = eligible.index(old)
        end = t0+model.DEADLINE+1
        packets = [(None, 0., 0.)]*t0
        for j, i in enumerate(eligible[t0:end], start=t0):
            if i is None:
                packets.append((None, 0., 0.))
            else:
                z = describe(x, easy, rows, j, t0, i)
                now, ahead = net(torch.from_numpy(z)).numpy()
                packets.append((i, now, ahead))
        for k in POLICIES:
            selected = model.stop(iter(packets), k)
            if selected is None:
                uses[k]["unused"] += 1
                continue
            t, i = selected
            if not t0 <= t < end or i == first:
                raise ValueError("Invalid support stopping action")
            pred[k][i] = p[i]
            uses[k].update(used=1, delayed=int(t > t0))
            if k != "immediate":
                pred[f"{k}_matched"][old] = p[old]
        at = np.flatnonzero(rec == r)
        pool = np.asarray([i for i in eligible[t0:end] if i is not None], dtype=int)
        candidate = cv[at].copy()
        candidate[np.searchsorted(at, pool)] = p[pool]
        j = base.best_second(cv[at], ols[at], candidate, y[at], valid[at], pred["gain"][at], cuts[0])
        pred["bounded_oracle"][old if j is None else at[j]] = p[old if j is None else at[j]]
        if j is not None:
            d0 = dict(y=y[at], valid=valid[at], rec=rec[at])
            exact = v.parent.score(d0, cv[at], ols[at], pred["bounded_oracle"][at], cuts)
            if exact["dual_positive"] != 1 or exact["edits"] != int(first is not None)+1:
                raise ValueError("Bounded witness failed original scoring")
    stats = {k: v.parent.score(d, cv, ols, z, cuts) for k, z in pred.items()}
    for k, old in (("immediate", "two_roles"), ("gain", "gain")):
        if any(stats[k][f] != ref["metrics"][old][f] for f in v.prior.FIELDS):
            raise ValueError("Immediate or first-slot reference changed")
    if not stats["immediate"]["dual_positive"] <= stats["bounded_oracle"]["dual_positive"] <= ceiling["any_query_joint"]:
        raise ValueError("Timing capacity outside its feasible bounds")
    for k in POLICIES:
        if (stats[k]["edits"]-stats["gain"]["edits"] != uses[k]["used"]
                or uses[k]["used"]+uses[k]["unused"] != timing["available"]
                or (k != "immediate" and stats[k]["edits"] != stats[f"{k}_matched"]["edits"])):
            raise ValueError("Timing budget or retrospective coverage changed")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ("gain", *POLICIES)]+[(k, f"{k}_matched") for k in POLICIES[1:]]
    pairs += [("continue", "myopic"), ("continue", "immediate")]
    effect = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass) for a, b in pairs}
    return dict(metrics=stats, effects=effect, uses=uses)


def execute(key, outer, old, source, gain_heads, ref, producers, release_heads, cap, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[:3] for s in v.parent.source.SITES}
    bank = {s: v.parent.prior.analytic(z[0]) for s, z in sites.items()}
    receipts = {(tuple(p["excluded"]), p["target"], p["seed"]): p for p in producers}
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
                h = release_heads["release_"+"_".join(pair)+f"_{seed}"]
                if set(h["identity"]["fit"]) & set(pair):
                    raise ValueError("Donor timing uses its own or outer labels")
                release = two.fit.load(h, lambda: hard.model.network(hard_width), hard.STEPS)
                path = np.load(run.checked(receipts[pair, s, seed]["prediction"]), allow_pickle=False)
                b = training(sites[s][0], bank[s][0], path, release, cuts([z for z in use if z != s])[0])
                if b is not None:
                    fit.append(b)
            n = fit[0][0].shape[1]
            name = f"{held}_{seed}"
            ident = dict(registration=key, held=held, fit=use, seed=seed)
            net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(n), STEPS,
                                               two.batcher(fit), model.loss, replay, beat)
            updates += u
            release = two.fit.load(outer[f"{held}_{seed}"], lambda: hard.model.network(hard_width), hard.STEPS)
            row = next(z for z in old["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(row["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, bank[held][0][:1], path[:1])[0].shape[1]
            gain_head = two.fit.load(gain_heads[name], lambda: prior.model.network(n), prior.STEPS)
            before = next(z for z in ref["by_site"][held] if z["seed"] == seed)
            c = next(z["counts"] for z in cap["by_site"][held] if z["seed"] == seed)
            out = evaluate(d, bank[held], path, release, gain_head, net, cuts(use), before, c)
            result[held].append(dict(seed=seed, donor_rows=sum(len(x) for x, _ in fit), **out))
            beat(phase="site", site=held, seed=seed, updates=updates, used=out["uses"]["continue"]["used"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in group]) for group in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]},
        uses={k: sum((r["uses"][k] for r in flat), Counter()) for k in POLICIES})
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
