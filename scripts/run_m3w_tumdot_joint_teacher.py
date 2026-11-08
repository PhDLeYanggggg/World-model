"""Compare donor-only joint and local teachers on the same TUMDOT roster."""

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
from scripts import audit_m3w_tumdot_three_reference as audit
from src.world_model import m3w_tumdot_joint_teacher as model

run, torch, study, trainer, INC = audit.run, audit.torch, audit.study, audit.trainer, audit.INC
parent, prior = audit.parent, audit.parent.prior
OUT = audit.OUT.parent/"tumdot_joint_teacher_v1"
PRIVATE = audit.PRIVATE.parent/OUT.name
STEPS = 400


@torch.inference_mode()
def score(rec, q, d, nets, cuts):
    cv, ols, bank, on, radius = (q[k] for k in ("cv", "ols", "bank", "on", "radius"))
    y, valid = d["future"], d["valid"]
    x = torch.from_numpy(q["x"].copy())
    x[..., -3:] = 0.
    hindcast = bank[:, INC]
    paths = dict(hindcast=hindcast)
    for k, net in nets.items():
        raw = net(x).numpy()
        for g in (False, True):
            a = model.choose(raw, INC, g)
            paths[k+("_guard" if g else "")] = bank[np.arange(len(bank)), a]
    metrics, contrasts = {}, {}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    for k, p in paths.items():
        np.testing.assert_array_equal((p != cv).any((1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1).max(1), radius, rtol=1e-7, atol=1e-13)
        metrics[k] = study.source.score(p, cv, ols, y, valid, rec, cuts, hindcast)
        metrics[k].update(edits=int(on.sum()), changed_from_hindcast=int((p != hindcast).any((1, 2)).sum()))
    pairs = [(k, "hindcast") for k in paths if k != "hindcast"] + [("joint", "local"), ("joint_guard", "local_guard")]
    for a, b in pairs:
        observed, lower = study.paired(paths[b], paths[a], y, valid)
        contrasts[a+"_vs_"+b] = dict(lower=float(100*lower.sum()/mass), observed=float(100*observed.sum()/mass))
    return dict(metrics=metrics, contrasts=contrasts)


def execute(replay, beat):
    ar, ap = audit.OUT/"registration.json", audit.OUT/"analysis.json"
    a = run.read(ap)
    if a["registration"] != run.sha(ar) or a["confirmation_access"] or not a["original_2pct"]:
        raise ValueError("Require frozen three-reference audit")
    key0, lineage, base, _ = prior.register()
    cache = prior.prepare(key0, lineage, base, beat)
    teachers = {f"{r}_{s}": run.artifact(audit.PRIVATE/f"{r}_{s}.json")
                for r in cache for s in study.parent.job.SEEDS}
    pr = parent.OUT/"analysis.json"
    previous = run.read(pr)
    reg = OUT/"registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(f) for f in (ar, ap, pr, Path(trainer.__file__))], teachers=teachers,
        steps=STEPS, original_2pct=True, confirmation_access=False, threads=4, workers=0,
        fit="four donors excluding held intersection and day and exact IDs; teacher labels from donors only",
        compare="joint maximin versus local best paired-hindcast imitation; identical donor data, features and loss",
        chooser="positive predicted pair gain; optional all/easy predicted harm guard; same radii and recipients",
        no_held_tuning=True, main_radius="half", quarter_is_sensitivity=True)))
    key = run.sha(reg)
    cuts = np.asarray(lineage[-1]["open_dd_positive_complete_cv_cuts"])
    trainer.PRIVATE = PRIVATE
    rows, heads, updates = {}, {}, 0
    for seed in study.parent.job.SEEDS:
        half = {r: prior.task(d, seed, .5) for r, d in cache.items()}
        for rec, d in cache.items():
            donors = prior.model.fit_roster(rec, cache)
            if len(donors) != 4 or any(not set(d["ids"]).isdisjoint(cache[r]["ids"]) for r in donors):
                raise ValueError("Donor overlap")
            data = {k: [] for k in ("local", "joint")}
            if not replay:
                for r in donors:
                    z = run.read(run.checked(teachers[f"{r}_{seed}"]))
                    if z["registration"] != a["registration"]:
                        raise ValueError("Changed teacher provenance")
                    joint = z["private_teacher"]["half"]
                    if joint is None or z["scores"]["half"]["oracle"]["status"] != "optimal":
                        raise ValueError("Require verified optimal fitting teacher")
                    q, t = half[r], cache[r]
                    on = q["on"]
                    y = parent.model.labels(q["cv"][on], q["bank"][on], t["future"][on], t["valid"][on],
                                             q["radius"][on], cuts[0], INC, study.paired)["pair_target"]
                    local = model.local_teacher(y[..., 0], INC)
                    for arm, target in (("local", local), ("joint", np.asarray(joint))):
                        if target.shape != local.shape:
                            raise ValueError("Misaligned teacher and active roster")
                        label = np.concatenate((y, np.eye(y.shape[1], dtype=np.float32)[target, :, None]), axis=-1)
                        data[arm].append((q["x"][on], label))
            nets = {}
            width = half[rec]["x"].shape[-1]
            for arm in data:
                name = f"{rec}_{seed}_{arm}"
                ident = dict(registration=key, held=rec, fit=list(donors), arm=arm, seed=seed)
                nets[arm], heads[name], n = trainer.fit(ident, name, seed, lambda: model.network(width), STEPS,
                    None if replay else prior.sampler(data[arm], False),
                    lambda net, x, y: model.loss(net, x, y, INC), replay, beat)
                updates += n
            values = {k: score(rec, half[rec] if k == "half" else prior.task(d, seed, f), d, nets, cuts)
                      for k, f in study.FRACTIONS.items()}
            old = next(v["scores"] for v in previous["rows"][rec] if v["seed"] == seed)
            if any(x["metrics"]["hindcast"] != old[k]["metrics"]["hindcast"] for k, x in values.items()):
                raise ValueError("Changed frozen comparator")
            rows.setdefault(rec, []).append(dict(seed=seed, fit=donors, scores=values))
            beat(phase="held", recording=rec, seed=seed, updates=updates)
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), rows=rows,
        summary=parent.aggregate(rows), original_2pct=True, confirmation_access=False,
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
