"""Run a causal peer alignment ablation on the fixed TUMDOT action menu."""

import argparse
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
from scripts import run_m3w_tumdot_incumbent_selector as parent
from src.evaluation.m3w_opendd_location import cohorts
from src.world_model import m3w_tumdot_peer_choice as model

prior = parent.prior
run, torch, study, trainer, INC = parent.run, parent.torch, parent.study, parent.trainer, parent.INC
OUT = prior.OUT.parent/"tumdot_peer_choice_v1"
PRIVATE = prior.PRIVATE.parent/OUT.name
ARMS = ("blind", "aligned", "permuted")
STEPS = 400


def register():
    a = prior.OUT.parent/"tumdot_joint_teacher_v1"
    paths = [a/"analysis.json", a/"registration.json", parent.OUT/"analysis.json", prior.OUT/"registration.json",
             Path(trainer.__file__)]
    previous = run.read(paths[0])
    if (previous["registration"] != run.sha(paths[1]) or previous["confirmation_access"]
            or not previous["original_2pct"] or not previous["source_is_previously_exposed"]):
        raise ValueError("Require frozen teacher ablation and original role")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in paths], original_2pct=True, confirmation_access=False,
        steps=STEPS, threads=4, workers=0, matched_recipients_and_radius=True,
        fit="four donors excluding held intersection and day; exact ID disjointness",
        arms="same architecture/draws; peer columns zeroed, aligned or cyclically permuted within current query",
        input="nearest past peers and candidate spacing versus their frozen causal hindcast; no future labels",
        target="direct paired hindcast lower gain and all/easy harm differences",
        primary="guarded aligned versus guarded blind and permuted at half radius; quarter sensitivity",
        no_held_tuning=True, campaigns=1)))
    return run.sha(f), run.read(paths[2])


def offsets_for(key, cache, beat):
    source = prior.source
    src = run.read(source.PUBLIC/"registration.json")["files"]
    out = {}
    for rec, (name, _) in source.ROSTER.items():
        p = PRIVATE/f"{rec}_alignment.json"
        if p.exists():
            value = run.read(p)
            if value["registration"] != key or value["source"] != src[rec]:
                raise ValueError("Changed alignment checkpoint")
            offsets = np.asarray(value["offsets"], int)
        else:
            raw = source.SOURCE/name
            if run.sha(raw) != src[rec]["sha256"]:
                raise ValueError("Changed publisher trajectory file")
            positions, _, _ = source.read_tracks(raw)
            queries = list(cohorts(np.isfinite(positions).all(2), 12))
            h = np.concatenate([positions[ids[:, None], start+12*np.arange(8)] for start, ids in queries])
            np.testing.assert_array_equal(h, cache[rec]["history"])
            starts = np.array([s for s, _ in queries], int)
            if np.any(np.diff(starts) <= 19*12):
                raise ValueError("Overlapping forecast windows")
            offsets = np.r_[0, np.cumsum([len(ids) for _, ids in queries])]
            run.once(p, run.stable(dict(registration=key, source=src[rec], offsets=offsets.tolist())))
        if offsets[0] != 0 or offsets[-1] != len(cache[rec]["history"]) or np.any(np.diff(offsets) < 2):
            raise ValueError("Invalid query offsets")
        out[rec] = offsets
        beat(phase="alignment", recording=rec, complete=len(out))
    if sum(len(v)-1 for v in out.values()) != 404:
        raise ValueError("Changed causal query roster")
    return out


def task(d, seed, f, offsets):
    z = prior.task(d, seed, f)
    z["x"][..., -3:] = 0.
    peer, shuf = model.contexts(d["history"], offsets, z["bank"], INC)
    z["features"] = {k: np.concatenate((z["x"], c), axis=-1) for k, c in
        (("blind", np.zeros_like(peer)), ("aligned", peer), ("permuted", peer[shuf]))}
    return z


@torch.inference_mode()
def score(rec, z, d, offsets, nets, cuts):
    cv, ols, bank, on = (z[k] for k in ("cv", "ols", "bank", "on"))
    y, valid = d["future"], d["valid"]
    paths = {"hindcast": bank[:, INC]}
    for a, net in nets.items():
        s = parent.model.centered(net, torch.from_numpy(z["features"][a]), INC).numpy()
        for g in (False, True):
            k = a+("_guard" if g else "")
            paths[k] = bank[np.arange(len(bank)), parent.model.choices(s, INC, g)]
    metrics, interaction = {}, {}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    for k, p in paths.items():
        np.testing.assert_array_equal((p != cv).any((1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1).max(1), z["radius"], rtol=1e-7, atol=1e-13)
        metrics[k] = study.source.score(p, cv, ols, y, valid, rec, cuts, paths["hindcast"])
        metrics[k].update(edits=int(on.sum()), changed_from_hindcast=int((p != paths["hindcast"]).any((1, 2)).sum()))
        error, n = model.pair_error(p, y, valid, d["history"], offsets)
        interaction[k] = dict(error=error, steps=n)
    pairs = [(k, "hindcast") for k in paths if k != "hindcast"]
    pairs += [("aligned"+g, b+g) for g in ("", "_guard") for b in ("blind", "permuted")]
    contrasts = {}
    for a, b in pairs:
        known, lower = study.paired(paths[b], paths[a], y, valid)
        contrasts[a+"_vs_"+b] = dict(lower=float(100*lower.sum()/mass), observed=float(100*known.sum()/mass))
    return dict(metrics=metrics, contrasts=contrasts, interaction=interaction)


def interactions(rows):
    result = {}
    for k in study.FRACTIONS:
        r = [x["scores"][k]["interaction"] for v in rows.values() for x in v]
        n = sum(x["hindcast"]["steps"] for x in r)
        if any(v["steps"] != x["hindcast"]["steps"] for x in r for v in x.values()) or n <= 0:
            raise ValueError("Pair metric must retain the same valid steps")
        result[k] = dict(steps=n, pair_step_ade={a: sum(x[a]["error"] for x in r)/n for a in r[0]})
    return result


def execute(replay, beat):
    key, previous = register()
    key0, lineage, base, _ = prior.register()
    cache = prior.prepare(key0, lineage, base, beat)
    offsets = offsets_for(key, cache, beat)
    cuts = np.asarray(lineage[-1]["open_dd_positive_complete_cv_cuts"])
    trainer.PRIVATE = PRIVATE
    rows, heads, updates = {}, {}, 0
    for seed in study.parent.job.SEEDS:
        half = {r: task(d, seed, .5, offsets[r]) for r, d in cache.items()}
        for rec, d in cache.items():
            fit = prior.model.fit_roster(rec, cache)
            if len(fit) != 4 or any(not set(d["ids"]).isdisjoint(cache[k]["ids"]) for k in fit):
                raise ValueError("Held source overlap")
            data = {k: [] for k in ARMS}
            if not replay:
                for r in fit:
                    q, f = half[r], cache[r]
                    on = q["on"]
                    y = parent.model.labels(q["cv"][on], q["bank"][on], f["future"][on], f["valid"][on],
                                             q["radius"][on], cuts[0], INC, study.paired)["pair_target"]
                    for k in ARMS:
                        data[k].append((q["features"][k][on], y))
            nets = {}
            width = half[rec]["features"]["blind"].shape[-1]
            for k in ARMS:
                name = f"{rec}_{seed}_{k}"
                identity = dict(registration=key, held=rec, fit=list(fit), seed=seed, arm=k)
                nets[k], heads[name], n = trainer.fit(identity, name, seed, lambda: prior.model.network(width),
                    STEPS, None if replay else prior.sampler(data[k], True),
                    lambda net, x, y: parent.model.loss(net, x, y, INC, True), replay, beat)
                updates += n
            z = {k: score(rec, half[rec] if k == "half" else task(d, seed, f, offsets[rec]),
                          d, offsets[rec], nets, cuts) for k, f in study.FRACTIONS.items()}
            old = next(x["scores"] for x in previous["rows"][rec] if x["seed"] == seed)
            if any(x["metrics"]["hindcast"] != old[k]["metrics"]["hindcast"] for k, x in z.items()):
                raise ValueError("Changed matched hindcast comparator")
            rows.setdefault(rec, []).append(dict(seed=seed, fit=fit, scores=z))
            beat(phase="held", recording=rec, seed=seed, updates=updates)
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), rows=rows,
        summary=parent.aggregate(rows), interactions=interactions(rows), original_2pct=True, confirmation_access=False,
        source_is_previously_exposed=True, no_deployment=True)))
    return updates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
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
        updates = execute(a.replay, beat)
        beat(complete=True, replay=a.replay, updates=updates)


if __name__ == "__main__":
    main()
