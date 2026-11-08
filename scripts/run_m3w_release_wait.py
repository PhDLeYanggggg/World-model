"""Train direct release-anchored timing value on fitting locations only."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_slot_timing as prior
from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_release_wait as model

base, parent, run, study, torch = prior.base, prior.parent, prior.run, prior.study, prior.torch
OUT, PRIVATE = prior.OUT.parent / "release_wait_v1", prior.PRIVATE.parent / "release_wait_v1"
STEPS = 800
ARMS = ("anchor", "prefix_defer", "predicted_gain", "predicted_gain_risk")


def register():
    key, a, ph, h, _, previous = prior.register()
    p = prior.OUT / "analysis.json"
    timing = run.read(p)
    if timing["registration"] != key or timing["confirmation_access"] or not timing["original_2pct"]:
        raise ValueError("Require a bound timing study")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        timing=run.artifact(p), obs_pred=[8, 12], seeds=study.prior.SEEDS, steps=STEPS, arms=ARMS,
        fitting="deterministic max-native-radius anchor per fitting query at three fixed amplitudes; no learned nominations",
        target="paired conservative gain and original risk increments of first positive prefix opportunity after release",
        matching="same released slots and original radii; preserve immediate anchors and their future credit",
        max_delay_queries=model.AGE, no_held_tuning=True, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, ph, h, previous


def training(d, cut):
    history, y, valid, q = (d[k] for k in ("h", "y", "valid", "query"))
    cv, ols, _, _ = prior.model.shape.rollout(history, 12)
    peak = prior.model.shape.peak(ols-cv)
    caps = np.bincount(q)
    offsets = np.r_[0, caps.cumsum()]
    anchors = [lo+peak[lo:hi].argmax() for lo, hi in zip(offsets[:-1], offsets[1:], strict=True) if hi-lo >= 2]
    anchors = np.asarray(anchors, dtype=int)
    anchors = anchors[peak[anchors] > 0.]
    output = []
    for f in (.25, .5, 1.):
        for at in np.array_split(anchors, max(1, (len(anchors)+255)//256)):
            if not len(at):
                continue
            amp = np.full(len(at), f)
            x, _, _, early, scale = prior.prior.model.features(history[at], amp)
            radii = peak[at]*f
            ar, ar_slot = [], []
            for j, i in enumerate(at):
                for t in range(q[i]+1, min(q[i]+model.AGE+1, len(caps))):
                    lo, hi = offsets[t:t+2]
                    if d["rec"][lo] != d["rec"][i] or caps[t] < 2:
                        continue
                    ok = np.arange(lo, hi)[peak[lo:hi] >= radii[j]]
                    ar.extend(ok)
                    ar_slot.extend([j]*len(ok))
            idx, slot = np.asarray(ar, dtype=int), np.asarray(ar_slot, dtype=int)
            pre = moments(cv[at], early, y[at], valid[at], cut)
            keys = ("all_excess", "easy_excess", "easy_signed")
            target = np.column_stack((paired(early, cv[at], y[at], valid[at])[1].mean(1), *[-pre[k] for k in keys]))
            if len(idx):
                _, path, s = prior.model.proposals(history[idx], radii[slot]/peak[idx])
                pos = np.flatnonzero(s > 0.)
                order = pos[np.lexsort((-s[pos], q[idx[pos]], slot[pos]))]
                _, first = np.unique(slot[order], return_index=True)
                pick = order[first]
                row, dest, to = idx[pick], path[pick], slot[pick]
                target[to, 0] += paired(cv[row], dest, y[row], valid[row])[1].mean(1)
                post = moments(cv[row], dest, y[row], valid[row], cut)
                target[to, 1:] += np.column_stack([post[k] for k in keys])
            output.append((x, (target/scale[:, None]).clip(-8., 8.).astype(np.float32)))
    return tuple(torch.from_numpy(np.concatenate(a)) for a in zip(*output, strict=True))


@torch.inference_mode()
def evaluate(d, bank, menu, count, rec, net, cuts):
    cv, ols = bank[:2]
    y, valid, query = (d[k] for k in ("y", "valid", "query"))
    pred = {k: cv.copy() for k in ARMS}
    actions = {k: Counter() for k in ARMS[1:]}
    for recid in np.unique(rec):
        at = np.flatnonzero(d["rec"] == recid)
        on = rec == recid
        qids = np.flatnonzero(on)
        q = query[at]-qids[0]
        cap = np.minimum(2, np.bincount(q)-1)
        eligible = menu["legal"][on, 1]
        use = np.arange(2) < count[on, None]
        initial = np.searchsorted(at, menu["row"][on][use])
        if not len(initial):
            continue
        amp = np.array([.25, .5, 1.])[menu["arm"][on][use]]
        rows, slots, _, paths, proxy, old, release, shift = prior.local_plan(
            cv[at], ols[at], d["h"][at], q, eligible, cap, initial, amp)
        x, _, _, ref, _ = prior.prior.model.features(d["h"][at[initial]], amp)
        np.testing.assert_array_equal(ref, paths[old])
        v = net(torch.from_numpy(x)).numpy()
        arms = dict(prefix_defer=proxy[old] < 0., predicted_gain=v[:, 0] > 0.,
                    predicted_gain_risk=(v[:, 0] > 0.) & (v[:, 1:] <= 0.).all(1))
        pred["anchor"][at[initial]] = ref
        for k, defer in arms.items():
            pick = model.schedule(q[rows], rows, slots, proxy, release, old, eligible, cap, defer)
            if (not np.isin(old[~defer], pick).all() or len(np.unique(slots[pick])) != len(pick)
                    or not prior.capacity.respects_tokens(np.bincount(q[rows[pick]], minlength=len(cap)), eligible)):
                raise ValueError("Wait decision violated the anchored reserve or spent an edit twice")
            pred[k][at[rows[pick]]] = paths[pick]
            actions[k].update(deferred=int(defer.sum()), moved=int((shift[pick] > 0).sum()), dropped=len(old)-len(pick))
    metrics = {k: parent.score(d, cv, ols, p, cuts) for k, p in pred.items()}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    direct = {k: float(100.*paired(pred["anchor"], pred[k], y, valid)[1].sum()/mass) for k in ARMS[1:]}
    return dict(metrics=metrics, direct=direct, actions=actions)


def execute(key, a, ph, h, previous, replay, beat):
    sites = {s: parent.prior.prior.old.load_site(s) for s in parent.source.SITES}
    rows, frozen, updates = {}, {}, 0
    for held, d in sites.items():
        donors = [s for s in sites if s != held]
        cuts = np.quantile(np.concatenate([sites[s]["cut_samples"] for s in donors]), [.25, .75])
        train = [training(sites[s], cuts[0]) for s in donors]
        width = train[0][0].shape[1]
        batch = prior.prior.batcher(train)
        b = parent.prior.analytic(d)
        file = parent.prior.prior.old.source.prior.PRIVATE / f"{held}_cohort.npz"
        with np.load(file, allow_pickle=False) as z:
            rec, starts = z["recording"], z["start"]
        rows[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            identity = dict(registration=key, held=held, donors=donors, seed=seed)
            head, frozen[name], u = study.fit(identity, name, seed, lambda: model.WaitValue(width), STEPS,
                batch, model.loss, replay, beat)
            updates += u
            rk = study.load(ph["heads"][f"{held}/{seed}/additive_rank"], lambda: study.prior.model.SwapValue(False), study.prior.STEPS)
            m = study.menus(d, b, rk)
            x, _, _ = base.features(d, b, m)
            net = study.load(h["heads"][f"{held}_{seed}_motion"], lambda: base.model.Value(x["motion"].shape[1]), base.STEPS)
            with torch.inference_mode():
                p = net(torch.from_numpy(x["motion"])).numpy()
            count = study.model.streaming_count(base.model.score(p, risk=False), m["legal"], rec, starts)
            v = evaluate(d, b, m, count, rec, head, cuts)
            old = next(r for r in previous["by_site"][held] if r["seed"] == seed)
            for k, x in v["metrics"]["anchor"].items():
                np.testing.assert_allclose(x, old["metrics"]["frozen_prefix"][k], atol=1e-10, rtol=0.)
            rows[held].append(dict(seed=seed, **v))
            beat(phase="held", held=held, seed=seed, actions=v["actions"])
    flat = [r for v in rows.values() for r in v]
    draw = np.random.default_rng(817).integers(len(rows), size=(3000, len(rows)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    totals = ("edits", "views", "undefined", "risk_pass", "dual_positive", "unknown_edits")
    summary = dict(arms={k: dict(vs_cv=ci(lambda r: r["metrics"][k]["vs_cv"]),
        **{f: sum(r["metrics"][k][f] for r in flat) for f in totals}) for k in ARMS},
        direct={k: ci(lambda r: r["direct"][k]) for k in ARMS[1:]},
        actions={k: sum((Counter(r["actions"][k]) for r in flat), Counter()) for k in ARMS[1:]})
    receipt = OUT / "head_freeze.json"
    run.once(receipt, run.stable(dict(registration=key, heads=frozen)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, summary=summary, heads=run.sha(receipt),
        by_site=rows, original_2pct=True, confirmation_access=False)))
    return updates


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
