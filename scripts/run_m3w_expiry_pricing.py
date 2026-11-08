"""Test expiry-aware forecasts without changing the released slots or decoder."""

import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import expit

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wait_peer_value as prior
from src.world_model import m3w_expiry_pricing as model

run, torch, parent, study, ledger = prior.run, prior.torch, prior.parent, prior.study, prior.prior
OUT, PRIVATE = prior.OUT.parent / "expiry_pricing_v1", prior.PRIVATE.parent / "expiry_pricing_v1"
STEPS = 800


def register():
    key, a, ph, h, pair, ranks, *_ = prior.register()
    files = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze", "expiry/registration", "expiry/analysis")]
    old, heads, er, e = [run.read(f) for f in files]
    if (any(v["registration"] != key for v in (old, heads)) or old["heads"] != run.sha(files[1])
            or e["registration"] != run.sha(files[2]) or er["parents"][0]["sha256"] != run.sha(files[0])
            or any(v["confirmation_access"] or not v["original_2pct"] for v in (old, er, e))):
        raise ValueError("Require the frozen peer and expiry evidence")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(f) for f in files], arms=model.ARMS, steps=STEPS, seeds=study.prior.SEEDS,
        native_steps=[8, 12], features="self-only causal release, ledger and opportunity history",
        comparison="one shared head, direct unconditional gain vs outcome mixture, downside price, and geometry-only expiry debit",
        threshold="same frozen first acceptance: joint >= .25, move >= .5, score > 0; at most one per recording",
        geometry="only the hypothetical expired edit is bounded by triangle inequality; no policy risk guarantee",
        calibration_or_confirmation=False, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, ph, h, pair, ranks, old


def withdrawal_geometry(cv, paths, rows, old, scale):
    return np.linalg.norm(paths[old]-cv[rows[old]], axis=-1).mean(1)/scale


def targets(cv, ols, y, valid, scale, paths, args, cuts):
    _, rows, _, _, _, old, _, _ = args
    target, _ = prior.labels(cv, ols, y, valid, scale, paths, args, cuts)
    r = rows[old]
    delta = ledger.prior.paired(paths[old], cv[r], y[r], valid[r])[1].mean(1)/scale
    bound = withdrawal_geometry(cv, paths, rows, old, scale)
    if (np.abs(delta) > bound+1e-10).any():
        raise ValueError("Withdrawal score violates its geometric envelope")
    drop = target[:, 0] == 0.
    np.testing.assert_allclose(target[drop, 3], delta[drop].clip(-8., 8.), atol=1e-7, rtol=1e-6)
    return np.c_[target, delta.clip(-8., 8.)].astype(np.float32)


def training(d, b, menu, count, rec, cuts):
    x, y = [], []
    for at, z, scale, paths, args in prior.packets(d, b, menu, count, rec):
        x.append(z[:, 0])
        y.append(targets(b[0][at], b[1][at], d["y"][at], d["valid"][at], scale, paths, args, cuts))
    return tuple(torch.from_numpy(np.concatenate(v)) for v in (x, y)) if x else None


@torch.inference_mode()
def evaluate(d, b, menu, count, rec, head, cuts):
    cv, ols = b[:2]
    y, valid = d["y"], d["valid"]
    pred = {k: cv.copy() for k in ("anchor", *model.ARMS)}
    match = {k: cv.copy() for k in model.ARMS}
    events = {k: Counter() for k in model.ARMS}
    site_mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    worst = {k: 0. for k in model.ARMS}
    for at, z, scale, paths, args in prior.packets(d, b, menu, count, rec):
        q, rows, slots, _, release, old, eligible, cap = args
        pred["anchor"][at[rows[old]]] = paths[old]
        values = head(torch.from_numpy(z[:, 0].copy())).numpy()
        p = expit(values[:, :3])
        distance = withdrawal_geometry(cv[at], paths, rows, old, scale)
        for k, s in model.score(values, distance).items():
            choice = prior.model.causal_pick(release, p[:, 2], p[:, 0], s)
            pick, _, _ = ledger.model.walk(*args, lambda a, _: a == choice if choice is not None else np.zeros(len(a), bool))
            counts = np.bincount(q[pick], minlength=len(cap))
            if ((counts > cap).any() or len(np.unique(slots[pick])) != len(pick)
                    or not ledger.timing.capacity.respects_tokens(counts, eligible)):
                raise ValueError("Expiry readout exceeded causal slot constraints")
            pred[k][at[rows[pick]]] = paths[pick]
            kept = old[slots[pick]]
            match[k][at[rows[kept]]] = paths[kept]
            drop = len(old)-len(pick)
            events[k].update(accepted=int(choice is not None), moved=int((q[pick] > release[slots[pick]]).sum()), expired=drop)
            if drop:
                e = old[choice]
                r = at[rows[e]]
                debit = ledger.prior.paired(paths[e:e+1], cv[r:r+1], y[r:r+1], valid[r:r+1])[1].sum()
                worst[k] = min(worst[k], float(100.*debit/site_mass))
    measure = {k: parent.score(d, cv, ols, v, cuts) for k, v in pred.items()}
    pair = lambda a, b: float(100.*ledger.prior.paired(b, a, y, valid)[1].sum()/site_mass)
    direct = {f"{k}_vs_anchor": pair(pred[k], pred["anchor"]) for k in model.ARMS}
    direct.update({f"{k}_vs_match": pair(pred[k], match[k]) for k in model.ARMS})
    direct.update({f"{k}_vs_direct": pair(pred[k], pred["direct"]) for k in model.ARMS if k != "direct"})
    expiry = {k: direct[f"{k}_vs_anchor"]-direct[f"{k}_vs_match"] for k in model.ARMS}
    return dict(metrics=measure, direct=direct, events=events, expiry_pp=expiry, worst_expiry_pp=worst)


def execute(key, a, ph, h, pair, ranks, old, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(d) for s, d in sites.items()}
    context = {}
    for s in sites:
        f = parent.prior.prior.old.source.prior.PRIVATE / f"{s}_cohort.npz"
        with np.load(f, allow_pickle=False) as z:
            context[s] = z["recording"], z["start"]
    rows, frozen, steps = {}, {}, 0

    def cuts(fit):
        return np.quantile(np.concatenate([sites[s]["cut_samples"] for s in fit]), [.25, .75])

    @lru_cache(maxsize=10)
    def donor_data(held, site, seed):
        omit = sorted((held, site))
        fit = [s for s in sites if s not in omit]
        tail = "_".join(omit)+f"_{seed}"
        rank, counter = ranks["rank_"+tail], pair["moments_"+tail]
        if any(z["identity"]["fit"] != fit or z["identity"]["seed"] != seed for z in (rank, counter)):
            raise ValueError("Donor release producer has held-site exposure")
        d, b = sites[site], banks[site]
        rec, start = context[site]
        m, count = ledger.frozen_count(d, b, rec, start, rank, counter, donor=True)
        return training(d, b, m, count, rec, cuts(fit))

    for held, d in sites.items():
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            donors = [s for s in sites if s != held]
            data = [x for s in donors if (x := donor_data(held, s, seed)) is not None]
            width = data[0][0].shape[-1]
            batch = ledger.timing.prior.batcher(data)
            ident = dict(registration=key, held=held, donors=donors, seed=seed)
            head, frozen[name], u = study.fit(ident, name, seed, lambda: model.Value(width), STEPS, batch, model.loss, replay, beat)
            steps += u
            rec, start = context[held]
            b = banks[held]
            m, count = ledger.frozen_count(d, b, rec, start, ph["heads"][f"{held}/{seed}/additive_rank"], h["heads"][name+"_motion"])
            v = evaluate(d, b, m, count, rec, head, cuts(donors))
            ref = next(r for r in old["by_site"][held] if r["seed"] == seed)
            for k, y in v["metrics"]["anchor"].items():
                np.testing.assert_allclose(y, ref["metrics"]["anchor"][k], atol=1e-10, rtol=0.)
            rows[held].append(dict(seed=seed, **v))
            beat(phase="held", held=held, seed=seed, events=v["events"])
    flat = [r for v in rows.values() for r in v]
    draw = np.random.default_rng(817).integers(len(rows), size=(3000, len(rows)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_location=float(x.min()))

    fields = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    summary = dict(arms={k: {f: sum(r["metrics"][k][f] for r in flat) for f in fields} for k in flat[0]["metrics"]},
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]},
        expiry={k: ci(lambda r: r["expiry_pp"][k]) for k in model.ARMS},
        worst_expiry_pp={k: min(r["worst_expiry_pp"][k] for r in flat) for k in model.ARMS},
        events={k: sum((Counter(r["events"][k]) for r in flat), Counter()) for k in model.ARMS})
    f = OUT / "head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=frozen)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary, heads=run.sha(f),
        by_site=rows, original_2pct=True, confirmation_access=False)))
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            event = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", event)
            if v["phase"] != "fit":
                print(event, flush=True)

        updates = execute(*register(), args.replay, beat)
        beat(phase="complete", updates=updates)


if __name__ == "__main__":
    main()
