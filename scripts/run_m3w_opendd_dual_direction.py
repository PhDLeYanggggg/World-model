"""Compare CV-only and dual-reference direction heads on fixed OpenDD edits."""

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
from scripts import audit_m3w_opendd_schedule_shapes as shape
from src.world_model import m3w_opendd_dual_direction as model

schedule = shape.schedule
record, base, stop, run, v, two, hard, prior = (
    schedule.record, schedule.base, schedule.stop, schedule.run, schedule.v, schedule.two, schedule.hard, schedule.prior)
OUT = schedule.OUT.parent/"opendd_dual_direction_v1"
PRIVATE = record.PRIVATE.parent/OUT.name
ARMS = ("neural", "ols_direction", "prefix_direction", "cv_head", "dual", "guard", "joint", "fixed_oracle")
STEPS = 400


def register():
    key, outer, old, source, gains, prev, _, _ = schedule.register()
    parents = [schedule.OUT/f"{k}.json" for k in ("analysis", "shape_registration", "shape_capacity")]
    a, r, b = map(run.read, parents)
    if (a["registration"] != key or b["registration"] != run.sha(parents[1])
            or r["parents"][0]["sha256"] != run.sha(parents[0]) or not b["outcome_aware_only"]
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (a, b, r))):
        raise ValueError("Require the verified schedule shape experiment")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(p) for p in parents],
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        primary="dual versus CV-only heads; exact same two-slot clock, recipients, counts and per-row peak",
        secondary="risk guard and same-query joint shape choice; analytic directions alone; exact fixed-row shape witness",
        target="candidate minus neural in conservative gain vs CV and OLS4, and in three CV risk excesses",
        fitting="double-excluded donor trajectories; same architecture, batches, seeds and steps, CV-only duplicates CV target",
        visibility="joint choice only on co-observed roles, no future second slot supplied",
        steps=STEPS, seeds=two.study.prior.SEEDS, threads=4, workers=0, native_steps=[8, 12],
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    _, _, _, _, _, _, producers, _, _ = schedule.admit.register()
    return run.sha(f), outer, old, source, gains, prev, producers


def bank(d, cv, ols, path, analytic):
    result = []
    for role, alpha in enumerate((base.model.GAIN_SCALE, base.model.SUPPORT_SCALE)):
        neural = cv+alpha*(path-cv)
        ps = (neural, shape.matched_direction(cv, neural, ols), shape.matched_direction(cv, neural, analytic))
        f, scale = zip(*(prior.feature(d, cv, p) for p in ps), strict=True)
        legal = np.stack([(p != cv).any((1, 2)) for p in ps], 1)
        z = np.stack(f, 1)
        c = np.broadcast_to(np.c_[np.eye(3), np.full(3, role)], (*legal.shape, 4))
        result.append((np.stack(ps, 1), np.concatenate((z, z-z[:, :1], c), 2).astype(np.float32),
                       legal, scale[0]))
    return result


def parts(d, cv, ols, paths, cut):
    values = [record.contributions(cv, ols, p, d["y"], d["valid"], cut) for p in paths.swapaxes(0, 1)]
    return tuple(np.stack(z, 1) for z in zip(*values, strict=True))


def training(d, cv, ols, path, analytic, cut):
    x, y = [], []
    for p, f, legal, scale in bank(d, cv, ols, path, analytic):
        gain, risk, *_ = parts(d, cv, ols, p, cut)
        z = np.concatenate((gain/12., risk), 2)
        target = ((z-z[:, :1])/scale[:, None, None]).clip(-8., 8.)
        x.append(f[legal])
        y.append(target[legal].astype(np.float32))
    return tuple(torch.from_numpy(np.concatenate(z)) for z in (x, y))


def exact(parts, legal, roles, full_easy, offset):
    values = []
    for z, ok, at in zip(parts, legal, roles, strict=True):
        if at is None:
            values.append([(0, np.zeros(2), np.zeros(3), 0., 0.)])
        else:
            gain, excess, ref, base, _ = z
            values.append([(k, gain[at, k], excess[at, k], ref[at, k], base[at, k])
                           for k in np.flatnonzero(ok[at])])
    best, choice = 0., None
    for k, g, r, e, b in values[0]:
        for j, g1, r1, e1, b1 in values[1]:
            lower = min(g[0]+g1[0], g[1]+g1[1]+offset)
            if (lower > best and b+b1 > 0. and e+e1 > 0. and full_easy > 0.
                    and ((r+r1) <= [0., 0., .02*full_easy]).all()):
                best, choice = lower, (k, j)
    return choice


@torch.inference_mode()
def evaluate(d, baseline, path, analytic, release, gain_head, models, cut, ref):
    cv, ols = baseline[:2]
    options = bank(d, cv, ols, path, analytic)
    truth = [parts(d, cv, ols, p, cut[0]) for p, _, _, _ in options]
    roster, *_ = record.menus(d, cv, path, release, gain_head)
    pred = {k: cv.copy() for k in ARMS}
    legal = [z[2] for z in options]
    q, rec, y, valid = (d[k] for k in ("query", "rec", "y", "valid"))
    usage = {k: Counter() for k in ARMS}
    same_query = 0
    expected_radius = np.zeros(len(cv))
    for r, a, b, _, _ in roster:
        rows = (a, b)
        picks = {k: [0, 0] for k in ARMS}
        score = []
        for i, (row, (p, x, ok, scale)) in enumerate(zip(rows, options, strict=True)):
            if row is None:
                score.append(None)
                continue
            expected_radius[row] = shape.shape.peak(p[row:row+1, 0]-cv[row:row+1])[0]
            for k, j in (("ols_direction", 1), ("prefix_direction", 2)):
                picks[k][i] = j if ok[row, j] else 0
            f = torch.from_numpy(x[row])
            z0, z = (m(f).numpy() for m in models)
            picks["cv_head"][i] = model.choose(z0, ok[row], "cv")
            picks["dual"][i] = model.choose(z, ok[row], "dual")
            picks["guard"][i] = picks["joint"][i] = model.choose(z, ok[row], "guard")
            score.append((z-z[0])*scale[row])
        if a is not None and b is not None and q[a] == q[b]:
            same_query += 1
            picks["joint"] = model.joint(*score, legal[0][a], legal[1][b])
        ix = np.flatnonzero(rec == r)
        full_easy = truth[0][2][ix, 0].sum()
        offset = truth[0][4][ix, 0].sum()
        oracle = exact(truth, legal, rows, full_easy, offset)
        if oracle is not None:
            picks["fixed_oracle"] = oracle
        for k, j in picks.items():
            for role, row in enumerate(rows):
                if row is not None:
                    pred[k][row] = options[role][0][row, j[role]]
                    usage[k][("neural", "ols", "prefix")[j[role]]] += 1
        if oracle is not None:
            score0 = v.parent.score(dict(y=y[ix], valid=valid[ix], rec=rec[ix]),
                                    cv[ix], ols[ix], pred["fixed_oracle"][ix], cut)
            if score0["dual_positive"] != 1:
                raise ValueError("Fixed-row oracle did not pass original scorer")
    stats = {k: v.parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
    for k, p in pred.items():
        np.testing.assert_allclose(shape.shape.peak(p-cv), expected_radius, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal((p != cv).any((1, 2)), expected_radius > 0.)
        if stats[k]["edits"] != ref["metrics"]["two_roles"]["edits"]:
            raise ValueError("Direction selection changed edit budget")
    for f in v.prior.FIELDS:
        if stats["neural"][f] != ref["metrics"]["two_roles"][f]:
            raise ValueError("Original shape and schedule did not replay")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+[(k, "neural") for k in ARMS[1:]]
    pairs += [("dual", "cv_head"), ("dual", "prefix_direction"), ("guard", "dual"), ("joint", "guard")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass)
               for a, b in pairs}
    easy = {}
    for k, p in pred.items():
        z = two.moments(cv, p, y, valid, cut[0])
        den = z["ref_easy"].sum()
        easy[k] = float(100.*z["easy_signed"].sum()/den) if den > 0. else None
    return dict(metrics=stats, effects=effects, directions=usage, same_query=same_query,
                easy_net_percent=easy, relative_joint_vs_guard=study.c.relative_effect(d, pred["guard"], pred["joint"]))


def execute(key, outer, old, source, gains, prev, producers, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    analytic = {s: shape.shape.analytic(d["h"])[2] for s, d in sites.items()}
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, frozen, updates = {}, {}, 0
    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit = []
            for s in use:
                p = record.read_prediction(refs, (held, s), s, seed)
                z = training(sites[s], *banks[s][:2], p, analytic[s], cuts([j for j in use if j != s])[0])
                if len(z[0]):
                    fit.append(z)
            n = fit[0][0].shape[1]
            name = f"{held}_{seed}"
            nets = []
            for mode in ("cv", "dual"):
                tag = f"{name}_{mode}"
                ident = dict(registration=key, held=held, fit=use, seed=seed, mode=mode)
                z = [(x, y[:, [0, 0, 2, 3, 4]]) for x, y in fit] if mode == "cv" else fit
                net, frozen[tag], u = two.fit.fit(ident, tag, seed, lambda: model.network(n), STEPS,
                                                  two.batcher(z), model.loss, replay, beat)
                updates += u
                nets.append(net)
            release = two.fit.load(outer[name], lambda: hard.model.network(hard_width), hard.STEPS)
            lookup = lambda z: next(a for a in z["by_site"][held] if a["seed"] == seed)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as data:
                p = data["comoving_ego"]
            cv = banks[held][0]
            n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], p[:1])[0].shape[1]
            g = two.fit.load(gains[name], lambda: prior.model.network(n), prior.STEPS)
            z = evaluate(d, banks[held], p, analytic[held], release, g, nets, cuts(use), lookup(prev))
            result[held].append(dict(seed=seed, donor_rows=sum(len(x) for x, _ in fit), **z))
            beat(phase="site", site=held, seed=seed, updates=updates, same_query=z["same_query"])
    flat = [r for group in result.values() for r in group]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in group]) for group in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]}, same_query=sum(r["same_query"] for r in flat),
        directions={k: sum((r["directions"][k] for r in flat), Counter()) for k in ARMS})
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=frozen)))
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
