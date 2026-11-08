"""Test whether causal recipient assignment restores support at identical query budgets."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_shape_menu_value as prior
from src.evaluation import m3w_recipient_slot_capacity as capacity
from src.world_model import m3w_recipient_slot as model

base, parent, run, torch, study = prior.base, prior.parent, prior.run, prior.torch, prior.study
OUT = prior.OUT.parent / "recipient_slot_v1"
PRIVATE = prior.PRIVATE.parent / OUT.name
STEPS = 800


def register():
    key, a, ph, h, _ = prior.register()
    p, f = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze")]
    old, heads = run.read(p), run.read(f)
    if (old["registration"] != key or heads["registration"] != key or old["heads"] != run.sha(f)
            or old["confirmation_access"] or not old["original_2pct"] or not old["exact_rows_and_radii"]):
        raise ValueError("Require frozen same-radius comparison and its original contract")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        capacity=run.artifact(Path(capacity.__file__)), prior=run.artifact(p), native_steps=[8, 12],
        arms=model.ARMS, amplitudes=model.AMPS, steps=STEPS, seeds=study.prior.SEEDS,
        matching="same query, number of edits, displacement radius per slot, observed-prefix shape rule",
        eligibility="recipient from the same query with sufficient OLS4 radius, at most one edit per row",
        training="six fitting locations; unselected rows at fixed amplitudes; no evaluated query outcomes",
        primary="joint chronological coverage versus frozen-prefix roster and joint-local control",
        proxy="product of predicted no-easy-reference probabilities; not a risk certificate",
        oracle_label_aware_only=True, no_threshold_tuning=True, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, ph, h, old


def prepare(d, cut):
    cv, ols, _, _ = model.shape.menu.rollout(d["h"], 12)
    on = model.shape.menu.peak(ols-cv) > 0.
    h, y, valid = [d[k][on] for k in ("h", "y", "valid")]
    data = []
    for amp in model.AMPS:
        x, cv, ols, path, scale = model.features(h, np.full(len(h), amp))
        data.append((x, model.target(cv, ols, path, y, valid, cut, scale)))
    return tuple(torch.from_numpy(np.concatenate(z)) for z in zip(*data, strict=True))


def batcher(train):
    def batch(rng):
        sites = np.bincount(rng.integers(len(train), size=128), minlength=len(train))
        pick = [rng.integers(len(d[0]), size=n) for d, n in zip(train, sites, strict=True)]
        return tuple(torch.cat([d[i][p] for d, p in zip(train, pick, strict=True)]) for i in range(2))
    return batch


def edges(d, cv, ols, menu, count):
    native_peak = model.shape.menu.peak(ols-cv)
    cut = np.r_[0, 1+np.flatnonzero(d["query"][1:] != d["query"][:-1]), len(cv)]
    np.testing.assert_array_equal(d["query"][cut[:-1]], np.arange(len(count)))
    edge_rows, edge_slots, amp, blocks = [], [], [], []
    for q in np.flatnonzero(count):
        n = count[q]
        initial = menu["row"][q, :n]
        initial_amp = np.array([.25, .5, 1.])[menu["arm"][q, :n]]
        radii = native_peak[initial]*initial_amp
        slot, row, a = model.feasible(np.arange(cut[q], cut[q+1]), radii, native_peak)
        old = np.flatnonzero(row == initial[slot])
        np.testing.assert_array_equal(slot[old], np.arange(n))
        a[old] = initial_amp
        lo = len(edge_rows)
        blocks.append((q, n, slice(lo, lo+len(row)), old, radii))
        edge_rows.extend(row)
        edge_slots.extend(2*q+slot)
        amp.extend(a)
    return np.array(edge_rows, dtype=int), np.array(edge_slots, dtype=int), np.asarray(amp), blocks


@torch.inference_mode()
def evaluate(d, bank, menu, count, rec, net, cuts):
    cv, ols = bank[:2]
    y, valid = d["y"], d["valid"]
    rows, slots, amp, blocks = edges(d, cv, ols, menu, count)
    x, c, o, path, scale = model.features(d["h"][rows], amp)
    np.testing.assert_array_equal(cv[rows], c)
    np.testing.assert_array_equal(ols[rows], o)
    z = net(torch.from_numpy(x)).numpy()
    p = expit(z[:, 5])
    v = z[:, :5]*scale[:, None]
    gain = v[:, :2].min(1)
    safe = gain - v[:, 2:].clip(min=0.)@np.array([2., 4., 1.])
    picked = {k: [] for k in model.ARMS}
    remaining = 1.
    last = None
    for q, n, use, anchor, radii in blocks:
        if rec[q] != last:
            remaining, last = 1., rec[q]
        r, s, prob = rows[use], slots[use] % 2, p[use]
        cov = .025*np.median(scale[use])
        on = dict(frozen_prefix=anchor,
            gain=model.assign(r, s, gain[use], prob, n),
            support=model.assign(r, s, np.zeros(len(r)), prob, n, 1.),
            joint_local=model.assign(r, s, safe[use], prob, n, cov),
            joint_prefix=model.assign(r, s, safe[use], prob, n, cov*remaining))
        for k, ix in on.items():
            a = use.start+ix
            np.testing.assert_array_equal(slots[a], 2*q+np.arange(n))
            np.testing.assert_allclose(model.shape.menu.peak(path[a]-cv[rows[a]]), radii, atol=1e-10, rtol=1e-8)
            picked[k].extend(a)
        remaining *= float(np.prod(1.-prob[on["joint_prefix"]]))
    pred, metrics = {}, {}
    anchor = np.asarray(picked["frozen_prefix"], dtype=int)
    for k, at in picked.items():
        a = np.asarray(at, dtype=int)
        if len(set(rows[a])) != len(a) or len(a) != count.sum():
            raise ValueError("Recipient assignment changed edit count or used a row twice")
        pred[k] = cv.copy()
        pred[k][rows[a]] = path[a]
        score = parent.score(d, cv, ols, pred[k], cuts)
        if score["edits"] != len(a):
            raise ValueError("A frozen slot did not produce a nonzero edit")
        metrics[k] = dict(**score, reassignments=int(np.sum(rows[anchor] != rows[a])))
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    direct = {f"{k}_vs_{ref}": float(100.*parent.transfer.paired(pred[ref], pred[k], y, valid)[1].sum()/mass)
        for k, ref in [(k, "frozen_prefix") for k in model.ARMS[1:]]+[("joint_prefix", "joint_local")]}
    oracle = Counter()
    for name in np.unique(d["rec"]):
        at = np.flatnonzero(d["rec"] == name)
        subset = d["rec"][rows] == name
        rr = np.searchsorted(at, rows[subset])
        status = capacity.assess(cv[at], ols[at], y[at], valid[at], rr, slots[subset], path[subset], cuts)
        oracle[status] += 1
    return dict(metrics=metrics, direct=direct, oracle=oracle)


def execute(key, a, ph, h, old, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    rows, frozen, steps = {}, {}, 0
    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        cuts = np.quantile(np.concatenate([sites[k]["cut_samples"] for k in donors]), [.25, .75])
        train = [prepare(sites[s], cuts[0]) for s in donors]
        width = train[0][0].shape[1]
        batch = batcher(train)
        b = parent.prior.analytic(d)
        file = parent.prior.prior.old.source.prior.PRIVATE / f"{held}_cohort.npz"
        with np.load(file, allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            identity = dict(registration=key, held=held, donors=donors, seed=seed)
            head, frozen[name], u = study.fit(identity, name, seed, lambda: model.Value(width), STEPS,
                batch, model.objective, replay, beat)
            steps += u
            rk = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            m = study.menus(d, b, rk)
            x, _, _ = base.features(d, b, m)
            net = study.load(h["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(x["motion"].shape[1]), base.STEPS)
            with torch.inference_mode():
                p = net(torch.from_numpy(x["motion"])).numpy()
            count = study.model.streaming_count(base.model.score(p, risk=False), m["legal"], rec, starts)
            val = evaluate(d, b, m, count, rec, head, cuts)
            ref = next(r for r in old["by_site"][held] if r["seed"] == seed)["metrics"]["full_prefix"]
            measured = val["metrics"]["frozen_prefix"]
            for k, v in ref.items():
                np.testing.assert_allclose(measured[k], v, atol=1e-10, rtol=0.)
            rows[held].append(dict(seed=seed, **val))
            beat(phase="held", held=held, seed=seed, oracle=val["oracle"])
    flat = [r for v in rows.values() for r in v]
    draw = np.random.default_rng(817).integers(len(rows), size=(3000, len(rows)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    totals = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits", "reassignments")
    result = dict(arms={k: dict(vs_cv=ci(lambda r: r["metrics"][k]["vs_cv"]),
        **{f: sum(r["metrics"][k][f] for r in flat) for f in totals}) for k in model.ARMS},
        direct={k: ci(lambda r: r["direct"][k]) for k in flat[0]["direct"]},
        oracle=sum((Counter(r["oracle"]) for r in flat), Counter()))
    hfile = OUT / "head_freeze.json"
    run.once(hfile, run.stable(dict(registration=key, heads=frozen)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(hfile), summary=result,
        by_site=rows, query_radii_exact=True, original_2pct=True, confirmation_access=False)))
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
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

        u = execute(*register(), a.replay, beat)
        beat(phase="complete", updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
