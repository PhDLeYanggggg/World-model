"""Train support-aware stopping for both roles on exposed OpenDD development."""

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
from scripts import run_m3w_opendd_dual_direction as direction
from src.world_model import m3w_opendd_dual_clock as model

record, base, stop, run, v, two, hard, prior = (
    direction.record, direction.base, direction.stop, direction.run, direction.v,
    direction.two, direction.hard, direction.prior)
OUT = direction.OUT.parent/"opendd_dual_clock_v1"
PRIVATE = record.PRIVATE.parent/OUT.name
STEPS = 600
MODES = ("myopic", "continuation")
ARMS = ("gain", "original", "myopic", "myopic_matched", "continuation", "continuation_matched")


def register():
    key, outer, old, source, gains, ref, producers = direction.register()
    rkey, _, _, _, _, _, _, releases, _, _ = record.register()
    parents = [direction.OUT/"analysis.json", direction.OUT/"head_freeze.json", record.OUT/"head_freeze.json"]
    a, h, b = map(run.read, parents)
    if (a["registration"] != key or a["heads"] != run.sha(parents[1]) or h["registration"] != key
            or b["registration"] != rkey or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require matched-direction and nested nomination lineage")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(p) for p in parents],
        steps=STEPS, seeds=two.study.prior.SEEDS, threads=4, workers=0, native_steps=[8, 12],
        fitting="double-excluded donor path and release head; nested first head excludes donor and outer; observed early-role contexts",
        target="whole-record finite lower gain vs CV and OLS4 with original risk checks; gated now increment, act-and-complete, act-minus-wait",
        future_queries_used_only_in_training_labels=model.LOOKAHEAD,
        nomination="first two current rows per role with old positive gain or easy >= .5",
        primary="continuation vs its retrospective earliest control at exact realized per-recording role counts",
        shape="same frozen neural direction and .25/.10 roles; realized radii can differ when rows change",
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), outer, old, source, gains, ref, producers, releases, b["first_heads"]


@torch.inference_mode()
def prepare(d, cv, path, release, first):
    sm, f, _, easy, blocks = stop.inputs(d, cv, path, release)
    big = cv+base.model.GAIN_SCALE*(path-cv)
    x, scale = prior.feature(d, cv, big)
    gain = first(torch.from_numpy(x)).numpy()[:, 0]*scale
    common = np.c_[easy, (gain/scale).clip(-8., 8.), np.log1p(scale)].astype(np.float32)
    features = (np.c_[x, common], np.c_[f, common])
    return (big, sm), features, blocks, gain, easy


def nominate(seq, gain, easy):
    def top(rows, score, valid):
        r = rows[valid]
        return tuple(map(int, r[np.argsort(-score[r], kind="stable")[:2]]))
    return [(top(z, gain, gain[z] > 0.), top(z, easy, easy[z] >= base.model.EASY_THRESHOLD)) for z in seq]


def record_value(components, at, scale):
    ref = components[0][2][at].sum()
    offset = components[0][4][at].sum()
    unit = max(float(np.median(scale[at])), 1e-8)

    def value(plan):
        if not plan:
            return 0.
        gain, risk, support, mass = np.zeros(2), np.zeros(3), 0., 0.
        for r, j in plan:
            g, p, e, b, _ = components[r]
            gain += g[j]
            risk += p[j]
            support += e[j]
            mass += b[j]
        lower = min(gain[0], gain[1]+offset)
        limit = np.array([0., 0., .02*ref])
        if support > 0. and ref > 0. and mass > 0. and (risk <= limit).all() and lower > 0.:
            return 1.+np.tanh(lower/(12.*unit))
        # Non-feasible plans never beat the zero decision value.
        return -.5-np.tanh((max(-lower, 0.)/12.+np.maximum(risk-limit, 0.).sum())/unit)
    return value


def continuations(packets, spent, first, last):
    yield spent
    for t in range(first, last):
        for a in model.proposals(packets[t], spent):
            if not a:
                continue
            p = spent+a
            if len(p) == 2:
                yield p
            else:
                yield from continuations(packets, p, t+1, last)


def training(d, cv, ols, path, release, first, cut):
    paths, features, blocks, gain, easy = prepare(d, cv, path, release, first)
    y, valid, rec = d["y"], d["valid"], d["rec"]
    comp = [record.contributions(cv, ols, p, y, valid, cut) for p in paths]
    _, scale = prior.feature(d, cv, paths[0])
    xs, zs, counts = [], [], Counter()
    for r, seq in blocks.items():
        packets = nominate(seq, gain, easy)
        value = record_value(comp, np.flatnonzero(rec == r), scale)
        prefix = {}
        for t, p in enumerate(packets):
            last = min(t+model.LOOKAHEAD+1, len(packets))
            for state in ((), *((z,) for z in prefix.values())):
                wait = max(value(z) for z in continuations(packets, state, t+1, last))
                for a in model.proposals(p, state):
                    if not a:
                        continue
                    after = state+a
                    now = min(value(after), value(after)-value(state))
                    complete = max(value(z) for z in continuations(packets, after, t+1, last))
                    xs.append(model.describe(features, t, p, state, a))
                    zs.append([now, complete, complete-wait])
                    counts.update(rows=1, positive_timing=int(complete > max(wait, 0.)))
            for k, z in enumerate(p):
                if z:
                    prefix.setdefault(k, (k, z[0]))
    if not xs:
        return None, counts
    return (torch.from_numpy(np.asarray(xs)), torch.from_numpy(np.asarray(zs, dtype=np.float32))), counts


@torch.inference_mode()
def evaluate(d, baseline, path, release, first, net, cuts, ref):
    cv, ols = baseline[:2]
    paths, features, blocks, gain, easy = prepare(d, cv, path, release, first)
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    pred = {k: cv.copy() for k in ARMS}
    diagnostics = Counter()
    comp = [record.contributions(cv, ols, p, y, valid, cuts[0]) for p in paths]
    _, scale = prior.feature(d, cv, paths[0])
    for r, seq in blocks.items():
        packets = nominate(seq, gain, easy)
        a, b, _ = base.model.choose((z, gain[z], easy[z]) for z in seq)
        if a is not None:
            pred["gain"][a] = pred["original"][a] = paths[0][a]
        if b is not None:
            pred["original"][b] = paths[1][b]
        def score(t, p, state, menu):
            x = np.stack([model.describe(features, t, p, state, u) for u in menu])
            return net(torch.from_numpy(x)).numpy()

        for mode in MODES:
            plan = model.stream(iter(packets), score, mode)
            roles = {k for k, _ in plan}
            control = model.earliest(packets, roles)
            if roles != {k for k, _ in control} or len(plan) != len(roles):
                raise ValueError("Realized role budget changed")
            for k, items in ((mode, plan), (mode+"_matched", control)):
                if len({j for _, j in items}) != len(items):
                    raise ValueError("Reused a recipient row")
                for role, j in items:
                    pred[k][j] = paths[role][j]
            clocks = dict(control)
            diagnostics.update({mode+"_gain": int(0 in roles), mode+"_support": int(1 in roles),
                                mode+"_both": int(len(roles) == 2),
                                mode+"_retimed": sum(q[j] != q[clocks[k]] for k, j in plan)})
        # Exact ceiling for the same causal nominations, labels never used by the stream.
        ix = np.flatnonzero(rec == r)
        value = record_value(comp, ix, scale)
        old_roles = {k for k, j in enumerate((a, b)) if j is not None}
        if old_roles:
            witness = next((z for z in continuations(packets, (), 0, len(packets))
                            if {k for k, _ in z} == old_roles and value(z) > 0.), None)
            if witness is not None:
                p = cv[ix].copy()
                for k, j in witness:
                    p[np.searchsorted(ix, j)] = paths[k][j]
                d0 = dict(y=y[ix], valid=valid[ix], rec=rec[ix])
                if v.parent.score(d0, cv[ix], ols[ix], p, cuts)["dual_positive"] != 1:
                    raise ValueError("Nominated witness failed original scoring")
                diagnostics["nominated_original_budget_capacity"] += 1
    metrics = {k: v.parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    for k, prev in (("gain", "gain"), ("original", "two_roles")):
        if any(metrics[k][f] != ref["metrics"][prev][f] for f in v.prior.FIELDS):
            raise ValueError("Frozen comparison did not reproduce")
    for k in MODES:
        if metrics[k]["edits"] != metrics[k+"_matched"]["edits"] or metrics[k]["edits"] != (
                diagnostics[k+"_gain"]+diagnostics[k+"_support"]):
            raise ValueError("Different scored intervention counts")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+[(k, k+"_matched") for k in MODES]
    pairs += [("continuation", "myopic"), ("continuation", "original")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass)
               for a, b in pairs}
    return dict(metrics=metrics, effects=effects, diagnostics=diagnostics)


def execute(key, outer, old, source, gains, ref, producers, releases, nested, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0
    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit, donors = [], Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                rr = releases["release_"+"_".join(pair)+f"_{seed}"]
                rr_first = nested[f"first_{held}_{s}_{seed}"]
                if (set(rr["identity"]["fit"]) & set(pair)
                        or set(rr_first["identity"]["excluded"]) != set(pair)
                        or set(rr_first["identity"]["fit"]) & set(pair)):
                    raise ValueError("Nomination model exposes outer or donor labels")
                release = two.fit.load(rr, lambda: hard.model.network(hard_width), hard.STEPS)
                path = record.read_prediction(refs, pair, s, seed)
                cv, ols = banks[s][:2]
                ds = sites[s]
                n = prior.feature({k: ds[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
                first = two.fit.load(rr_first, lambda: prior.model.network(n), prior.STEPS)
                z, info = training(ds, cv, ols, path, release, first, cuts([k for k in use if k != s])[0])
                donors.update(info)
                if z is not None:
                    fit.append(z)
            name = f"{held}_{seed}"
            n = fit[0][0].shape[1]
            ident = dict(registration=key, held=held, fit=use, seed=seed)
            net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(n), STEPS,
                                                two.batcher(fit), model.loss, replay, beat)
            updates += u
            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            release = two.fit.load(outer[name], lambda: hard.model.network(hard_width), hard.STEPS)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            cv = banks[held][0]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
            first = two.fit.load(gains[name], lambda: prior.model.network(n), prior.STEPS)
            z = evaluate(d, banks[held], path, release, first, net, cuts(use), lookup(ref))
            result[held].append(dict(seed=seed, donor=donors, **z))
            beat(phase="site", site=held, seed=seed, updates=updates,
                 n=donors["rows"], positive=donors["positive_timing"])
    flat = [r for group in result.values() for r in group]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in z]) for z in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]},
        diagnostics=sum((r["diagnostics"] for r in flat), Counter()))
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
            if p.get("phase") != "fit":
                print(z, flush=True)

        two.fit.PRIVATE = PRIVATE
        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
