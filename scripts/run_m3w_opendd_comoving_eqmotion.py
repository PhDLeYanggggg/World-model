"""Fit a published-family control and a comoving-frame forecast on exposed OpenDD."""

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
from scripts import run_m3w_donor_shape_choice as prior
from scripts import run_m3w_donor_shape_ceiling as ceiling
from src.world_model import m3w_opendd_comoving_eqmotion as model
from src.world_model import m3w_eqmotion_adapter as author
from src.world_model import m3w_donor_motion_shape as radius

run, parent, paired, ledger, two = prior.run, prior.parent, prior.paired, prior.ledger, prior.two
OUT = prior.OUT.parent / "opendd_comoving_eqmotion_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS, BATCH = 800, 32


def register():
    key, a, ph, h, _, source = prior.register()
    p = prior.OUT / "analysis.json"
    old = run.read(p)
    if old["registration"] != key or not old["original_2pct"] or old["confirmation_access"]:
        raise ValueError("Require the bound fixed-slot comparison")
    _, upstream = author.verified_source()
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), capacity=run.artifact(Path(ceiling.__file__)),
        parent=run.artifact(p), author=upstream, architecture=model.ARCH, arms=model.ARMS,
        seeds=two.study.prior.SEEDS, native_steps=[8, 12], steps=STEPS, batch=BATCH, threads=4, workers=0,
        fit="site-balanced observed-label training rows; common draws and normalized robust path loss",
        compare="centered peers vs comoving peers, then remove peers while retaining common scale and capacity",
        primary="full-roster direct lower gain over CV and OLS4; comoving vs centered at identical budget",
        matched="force matched neural shapes at the original slots/radii; retain original on zero neural offset",
        ceiling="two shapes (original, neural), exact original risk and both references at fixed slots",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"),
        original_2pct=True, confirmation_access=False, model_selection=False)))
    return run.sha(reg), a, ph, h, old, source


def load(s, source):
    d = parent.prior.prior.old.load_site(s)
    with np.load(source.PRIVATE / f"{s}_cohort.npz", allow_pickle=False) as z:
        offsets, rec, starts = (z[k] for k in ("offsets", "recording", "start"))
    np.testing.assert_array_equal(d["query"], np.repeat(np.arange(len(rec)), np.diff(offsets)))
    x = model.context(d["h"], offsets)
    y = torch.from_numpy(np.where(d["valid"][..., None], d["y"], 0.).astype(np.float32))
    valid = torch.from_numpy(d["valid"])
    return d, rec, starts, x, y, valid


def batcher(data):
    eligible = [np.flatnonzero(d[-1].numpy().any(1)) for d in data]
    if any(not len(p) for p in eligible):
        raise ValueError("Require observed training targets in every fitting site")

    def sample(rng):
        sizes = rng.multinomial(BATCH, np.full(len(data), 1./len(data)))
        rows = [r[rng.integers(len(r), size=n)] for r, n in zip(eligible, sizes, strict=True)]
        x = tuple(torch.cat([d[3][j][i] for d, i in zip(data, rows, strict=True)]) for j in range(4))
        y, valid = (torch.cat([d[j][i] for d, i in zip(data, rows, strict=True)]) for j in (4, 5))
        return x, y, valid

    return sample


@torch.inference_mode()
def predict(net, x):
    p = np.concatenate([net(*(v[i:i+128] for v in x[:3])).numpy() for i in range(0, len(x[0]), 128)])
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite trajectory prediction")
    return p.astype(float)


def freeze_paths(name, paths):
    f = PRIVATE / f"{name}_paths.npz"
    if f.exists():
        with np.load(f, allow_pickle=False) as z:
            for k, p in paths.items():
                np.testing.assert_array_equal(z[k], p)
    else:
        temp = f.with_suffix(".pending.npz")
        np.savez(temp, **paths)
        temp.replace(f)
    return run.artifact(f)


def raw_score(d, bank, p, cuts):
    cv, ols = bank[:2]
    y, valid = d["y"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    full, end = valid.all(1), valid[:, -1]
    refs = dict(cv=cv, ols4=ols)
    paths = {**refs, **p}
    out = {}
    for k, v in paths.items():
        err = np.linalg.norm(v-y, axis=-1)
        m = two.moments(cv, v, y, valid, cuts[0])
        easy_mass = m["ref_easy"].sum()
        lower = {r: float(100.*paired(a, v, y, valid)[1].sum()/mass) for r, a in refs.items()}
        known = {r: float(100.*paired(a, v, y, valid)[0].sum()/mass) for r, a in refs.items()}
        out[k] = dict(complete_ade_sum=float(err[full].sum()/12.), endpoint_fde_sum=float(err[end, -1].sum()),
            p95_complete_ade=float(np.quantile(err[full].mean(1), .95)), lower=lower, known=known,
            easy_net_percent=float(100.*m["easy_signed"].sum()/easy_mass) if easy_mass > 0. else None)
    effects = {f"comoving_peers_vs_{k}": float(100.*paired(p[k], p["comoving_peers"], y, valid)[1].sum()/mass)
               for k in ("centered_peers", "comoving_ego")}
    return dict(complete=int(full.sum()), endpoint=int(end.sum()), arms=out, effects=effects)


def matched(d, bank, menu, count, packet, pred, cuts, ref):
    cv, ols = bank[:2]
    use = np.arange(2) < count[:, None]
    at, arm = menu["row"][use], menu["arm"][use]
    anchor, prefix = packet["paths"][at, arm, 0], packet["paths"][at, arm, 3]
    effects, result, oracle = {}, {}, {}
    y, valid = d["y"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    frozen = cv.copy()
    frozen[at] = anchor
    base = {"anchor": anchor, "prefix_match": prefix}
    target_radius = radius.peak(anchor-cv[at])
    for k, p in pred.items():
        base[k] = radius.match_radius(cv[at], anchor, (p[at]-cv[at])[:, None])[:, 0]
    for k, p in base.items():
        f = cv.copy()
        f[at] = p
        np.testing.assert_allclose(radius.peak(f[at]-cv[at]), target_radius, atol=1e-10)
        np.testing.assert_array_equal(np.flatnonzero((f != cv).any((1, 2))), np.sort(at))
        result[k] = parent.score(d, cv, ols, f, cuts)
        if k in ("anchor", "prefix_match"):
            for name, val in ref["policies"][k].items():
                np.testing.assert_allclose(result[k][name], val, atol=1e-10, rtol=0.)
        else:
            effects[k] = float(100.*paired(frozen, f, y, valid)[1].sum()/mass)
            bank_for_ceiling = np.zeros((*bank[-1].shape, 2, 12, 2))
            bank_for_ceiling[at, arm, 0], bank_for_ceiling[at, arm, 1] = anchor, p
            oracle[k] = ceiling.count(d, bank, {"paths": bank_for_ceiling}, menu, count, cuts)
    return dict(policies=result, effects_vs_anchor=effects, oracle=oracle)


def execute(key, a, ph, h, old, source, replay, beat):
    sites = {s: load(s, source) for s in parent.source.SITES}
    result, heads, updates = {}, {}, 0
    for held, (d, rec, starts, x, _, _) in sites.items():
        donors = [s for s in sites if s != held]
        cuts = np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in donors]), [.25, .75])
        sample = batcher([sites[s] for s in donors])
        bank = parent.prior.analytic(d)
        packet = prior.prepare({s: v[0] for s, v in sites.items()}, held, donors, bank)
        result[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            paths = {}
            for k in model.ARMS:
                name = f"{held}_{seed}_{k}"
                ident = dict(registration=key, held=held, fit=donors, seed=seed, arm=k)
                net, heads[name], u = two.fit.fit(ident, name, seed, lambda: model.Forecast(k),
                                                 STEPS, sample, model.loss, replay, beat)
                updates += u
                paths[k] = predict(net, x)
            receipt = freeze_paths(f"{held}_{seed}", paths)
            m, n = ledger.frozen_count(d, bank, rec, starts, ph["heads"][f"{held}/{seed}/additive_rank"],
                                      h["heads"][f"{held}_{seed}_motion"])
            ref = next(v for v in old["by_site"][held] if v["seed"] == seed)
            result[held].append(dict(seed=seed, prediction=receipt, raw=raw_score(d, bank, paths, cuts),
                matched=matched(d, bank, m, n, packet, paths, cuts, ref)))
            beat(phase="readout", held=held, seed=seed, updates=updates)

    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def interval(fn):
        x = np.array([np.mean([fn(r) for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    raw = {k: dict(complete_ade=sum(v["raw"]["arms"][k]["complete_ade_sum"] for v in flat)/sum(v["raw"]["complete"] for v in flat),
        endpoint_fde=sum(v["raw"]["arms"][k]["endpoint_fde_sum"] for v in flat)/sum(v["raw"]["endpoint"] for v in flat),
        lower={b: interval(lambda v: v["raw"]["arms"][k]["lower"][b]) for b in ("cv", "ols4")},
        known={b: interval(lambda v: v["raw"]["arms"][k]["known"][b]) for b in ("cv", "ols4")})
        for k in flat[0]["raw"]["arms"]}
    totals = {k: {field: sum(r["matched"]["policies"][k][field] for r in flat)
                 for field in ("edits", "views", "undefined", "risk_pass", "dual_positive")}
              for k in flat[0]["matched"]["policies"]}
    capacity = {k: sum((Counter(r["matched"]["oracle"][k]) for r in flat), Counter()) for k in model.ARMS}
    effects = {k: interval(lambda r: r["raw"]["effects"][k]) for k in flat[0]["raw"]["effects"]}
    matched_effects = {k: interval(lambda r: r["matched"]["effects_vs_anchor"][k]) for k in model.ARMS}
    freeze = OUT / "head_freeze.json"
    run.once(freeze, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(freeze), by_site=result,
        summary=dict(raw=raw, raw_effects=effects, matched=totals, matched_effects=matched_effects, capacity=capacity),
        original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    two.fit.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", z)
            if v.get("step", STEPS) == STEPS:
                print(z, flush=True)

        u = execute(*register(), a.replay, beat)
        beat(phase="complete", updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
