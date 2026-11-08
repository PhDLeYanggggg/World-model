"""Fit prospective early support against the complete original recording policy."""

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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The historical runners import their receipt ancestry through a deep chain.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
from scripts import run_m3w_opendd_early_reference as early

run, record, v, two, hard, prior = early.run, early.record, early.v, early.two, early.hard, early.prior
wait, model = early.wait, early.model
OUT = early.OUT.parent/"opendd_early_advantage_v1"
PRIVATE = early.PRIVATE.parent/OUT.name
STEPS = 500
QUANTILE = .90


def register():
    key, *inputs = early.register()
    a = early.timing.legacy.read(early.OUT.name)
    if a["registration"] != key:
        raise ValueError("Require frozen local reference experiment")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(early.OUT/f"{k}.json")
            for k in ("registration", "analysis", "head_freeze")],
        primary="prospective recording advantage versus original and same-query, same-radius easy ranking",
        arms="mass=recording utility plus feasibility transitions; incidence=utility-only; shared training",
        target="replacement vs complete original: conservative utility plus joint/risk/reference transitions",
        gate="strictly above max(0, donor-record maximum-score 90th percentile); no held labels or scores",
        donor_gate_in_sample=True, independent_calibration=False, seeds=two.study.prior.SEEDS,
        steps=STEPS, threads=4, workers=0, native_steps=[8, 12],
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), *inputs


def data(d, cv, ols, cut, pack):
    large, small, blocks, gain, easy, radius, x, scale = pack
    rec, q, y, valid = (d[k] for k in ("rec", "query", "y", "valid"))
    a = record.contributions(cv, ols, small, y, valid, cut)
    b = record.contributions(cv, ols, large, y, valid, cut)
    add = v.paired(cv, small, y, valid)[1].sum(1)
    remove = v.paired(small, cv, y, valid)[1].sum(1)
    features, targets, groups, count = [], [], [], Counter()
    for r, seq in blocks.items():
        g, s, _ = wait.clock.base.model.choose((ix, gain[ix], easy[ix]) for ix in seq)
        at = np.flatnonzero(rec == r)
        legal = (radius[at] > 0.) & (at != g)
        if g is not None:
            legal &= q[at] <= q[g]
        if s is not None:
            legal &= q[at] < q[s]
        use = at[legal]
        if not len(use):
            continue
        fixed = tuple(t[g] for t in b[:4]) if g is not None else None
        easy_total, ols_delta = a[2][at].sum(), a[4][at].sum()
        after_risk, after_joint = early.support_targets(*(t[use] for t in a[:4]), easy_total, fixed, ols_delta)
        if s is None:
            zero = [np.zeros_like(t[:1]) for t in a[:4]]
            before_risk, before_joint = early.support_targets(*zero, easy_total, fixed, ols_delta)
        else:
            before_risk, before_joint = early.support_targets(*(t[s:s+1] for t in a[:4]), easy_total, fixed, ols_delta)
        if g is not None and b[2][g] > 0.:
            lost = np.zeros(len(use), dtype=bool)
        else:
            lost = (s is not None and a[2][s] > 0.) & (a[2][use] <= 0.)
        util = ((add[use]+(remove[s] if s is not None else 0.))/(12.*scale[use])).clip(-1., 1.)
        outcome = (after_joint.astype(float)-before_joint.astype(float)
                   -.5*(before_risk & ~after_risk)-.5*lost+.1*util)
        features.append(x[use])
        targets.append(np.c_[outcome, util].astype(np.float32))
        groups.append(len(use))
        count.update(rows=len(use), records=1, favorable=int((outcome > 0.).sum()))
    if not groups:
        return None, [], count
    return (torch.from_numpy(np.concatenate(features)), torch.from_numpy(np.concatenate(targets))), groups, count


def loss(net, x, y):
    return F.smooth_l1_loss(net(x), y, beta=.1)


@torch.inference_mode()
def thresholds(net, donors):
    maxima = []
    for (x, _), sizes in donors:
        z = net(x).numpy()
        maxima.extend(np.maximum.reduceat(z, np.r_[0, np.cumsum(sizes)[:-1]], axis=0))
    return np.maximum(0., np.quantile(np.stack(maxima), QUANTILE, axis=0))


def execute(key, outer, old, source, gains, _rank, producers, releases, nested, _protected, ref, replay, beat):
    sites = {s: wait.study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0

    def pack(s, path, r, f):
        d, cv = sites[s], banks[s][0]
        release = two.fit.load(r, lambda: hard.model.network(hard_width), hard.STEPS)
        n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
        return early.prepare(d, cv, path, release, two.fit.load(f, lambda: prior.model.network(n), prior.STEPS))

    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            donors, counts = [], Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                r = releases["release_"+"_".join(pair)+f"_{seed}"]
                f = nested[f"first_{held}_{s}_{seed}"]
                if (set(r["identity"]["fit"]) & set(pair) or set(f["identity"]["excluded"]) != set(pair)
                        or set(f["identity"]["fit"]) & set(pair)):
                    raise ValueError("Nomination heads expose held or donor labels")
                path = record.read_prediction(refs, pair, s, seed)
                fit, group, info = data(sites[s], *banks[s][:2], cuts([t for t in use if t != s])[0], pack(s, path, r, f))
                if fit is not None:
                    donors.append((fit, group))
                counts.update(info)
            name = f"{held}_{seed}"
            n = donors[0][0][0].shape[1]
            ident = dict(registration=key, held=held, fit=use, seed=seed)
            net, freeze[name], u = two.fit.fit(ident, name, seed, lambda: model.network(n), STEPS,
                two.batcher([z for z, _ in donors]), loss, replay, beat)
            updates += u
            gate = thresholds(net, donors)

            def readout(x, i):
                score = net(x)[:, i]-float(gate[i])
                return torch.stack((score, torch.ones_like(score)), dim=1)

            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                p = pack(held, z["comoving_ego"], outer[name], gains[name])
            z = early.evaluate(d, *banks[held][:2], cuts(use), p,
                               dict(mass=lambda x: readout(x, 0), incidence=lambda x: readout(x, 1)), lookup(ref))
            result[held].append(dict(seed=seed, donor=counts, donor_threshold=gate.tolist(), **z))
            beat(phase="site", site=held, seed=seed, updates=updates, records=counts["records"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in early.ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]}, diagnostics=sum((r["diagnostics"] for r in flat), Counter()))
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
