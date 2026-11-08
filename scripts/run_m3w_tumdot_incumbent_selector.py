"""Learn direct hindcast replacement on the matched TUMDOT menu."""

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
from scripts import run_m3w_tumdot_matched_selector as prior
from src.world_model import m3w_tumdot_incumbent_selector as model

run, torch, study, trainer = prior.run, prior.torch, prior.study, prior.trainer
OUT = prior.OUT.parent/"tumdot_incumbent_selector_v1"
PRIVATE = prior.PRIVATE.parent/OUT.name
INC = study.ARMS.index("hindcast")
STEPS = 400


@torch.inference_mode()
def score(rec, z, d, nets, cuts):
    cv, ols, bank, on = (z[k] for k in ("cv", "ols", "bank", "on"))
    x = torch.from_numpy(z["x"].copy())
    x[..., -3:] = 0.
    value = {k: model.centered(net, x, INC).numpy() for k, net in nets.items()}
    use = {k: model.choices(s, INC) for k, s in value.items()}
    use["pair_guard"] = model.choices(value["pair_target"], INC, True)
    use["hindcast"] = np.full(len(bank), INC, dtype=int)
    paths = {k: bank[np.arange(len(bank)), i] for k, i in use.items()}
    y, valid = d["future"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    metrics, contrasts = {}, {}
    for k, p in paths.items():
        np.testing.assert_array_equal(np.any(p != cv, axis=(1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1).max(1), z["radius"], atol=1e-13, rtol=1e-7)
        metrics[k] = study.source.score(p, cv, ols, y, valid, rec, cuts, paths["hindcast"])
        metrics[k]["changed_from_hindcast"] = int(np.any(p != paths["hindcast"], axis=(1, 2)).sum())
        metrics[k]["edits"] = int(on.sum())
    for a, b in (("pair_target", "hindcast"), ("cv_target", "hindcast"), ("pair_guard", "hindcast"),
                 ("pair_target", "cv_target"), ("pair_guard", "pair_target")):
        known, lower = study.paired(paths[b], paths[a], y, valid)
        contrasts[a+"_vs_"+b] = dict(lower=float(100*lower.sum()/mass), observed=float(100*known.sum()/mass))
    return dict(metrics=metrics, contrasts=contrasts)


def aggregate(rows):
    groups = sorted({k.split("-")[0] for k in rows})
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    result = {}
    for part in study.FRACTIONS:
        grouped = [[r["scores"][part] for k, z in rows.items() if k.startswith(g+"-") for r in z] for g in groups]
        flat = [r for z in grouped for r in z]
        if len(flat) != 27:
            raise ValueError("Incomplete cohort")

        def ci(fn):
            v = np.asarray([np.mean([fn(r) for r in z]) for z in grouped])
            return dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist(),
                        by_intersection=dict(zip(groups, v.tolist(), strict=True)))

        metrics = {}
        for a in flat[0]["metrics"]:
            r = [x["metrics"][a] for x in flat]
            metrics[a] = dict(risk_pass=sum(m["risk"]["supported"] for m in r),
                joint_cv=sum(m["risk"]["supported"] and m["lower_pp"]["cv"] > 0. for m in r),
                joint_cv_ols=sum(m["risk"]["supported"] and min(m["lower_pp"]["cv"], m["lower_pp"]["ols4"]) > 0. for m in r),
                edits=sum(m["edits"] for m in r), changed_from_hindcast=sum(m["changed_from_hindcast"] for m in r),
                worst_easy_pct=max(100*m["risk"]["known_easy_net_degradation"] for m in r),
                ade=float(np.mean([m["available_ade"] for m in r])),
                fde=float(np.mean([m["available_fde"] for m in r])),
                cv=ci(lambda x: x["metrics"][a]["lower_pp"]["cv"]),
                ols4=ci(lambda x: x["metrics"][a]["lower_pp"]["ols4"]))
        effects = {k: {f: ci(lambda z: z["contrasts"][k][f]) for f in ("lower", "observed")}
                   for k in flat[0]["contrasts"]}
        result[part] = dict(metrics=metrics, effects=effects, views=len(flat))
    return result


def execute(replay, beat):
    key0, lineage, previous, _ = prior.register()
    a, r, h = (prior.OUT/f"{k}.json" for k in ("analysis", "registration", "head_freeze"))
    ref = run.read(a)
    if ref["registration"] != key0 or ref["heads"] != run.sha(h) or ref["confirmation_access"]:
        raise ValueError("Require frozen first selector trial")
    reg = OUT/"registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in (a, r, h, Path(trainer.__file__))],
        original_2pct=True, confirmation_access=False, steps=STEPS, threads=4, workers=0,
        objective="replace CV lower target with exact paired hindcast lower target; same features, donors and MSE",
        features="without past replay scores for both arms; same learned harm targets",
        compare="direct paired lower and recording risk, not differences of lower scores",
        rule="only switch on predicted positive centered gain; guard also requires non-increased predicted all/easy harm",
        matched_recipients_and_radius=True, no_held_tuning=True, campaigns=1)))
    key = run.sha(reg)
    cache = prior.prepare(key0, lineage, previous, beat)
    cuts = np.asarray(lineage[-1]["open_dd_positive_complete_cv_cuts"])
    trainer.PRIVATE = PRIVATE
    rows, heads, updates = {}, {}, 0
    for seed in study.parent.job.SEEDS:
        half = {k: prior.task(v, seed, .5) for k, v in cache.items()}
        fit_data = {k: {} for k in cache}
        for rec, d in cache.items():
            donors = prior.model.fit_roster(rec, cache)
            if len(donors) != 4 or any(not set(d["ids"]).isdisjoint(cache[k]["ids"]) for k in donors):
                raise ValueError("Training overlap")
            for k in donors:
                if k in fit_data[rec] or replay:
                    continue
                q, z = half[k], cache[k]
                m = q["on"]
                fit_data[rec][k] = model.labels(q["cv"][m], q["bank"][m], z["future"][m], z["valid"][m],
                                                q["radius"][m], cuts[0], INC, study.paired)
            nets = {}
            width = half[rec]["x"].shape[-1]
            for arm in ("cv_target", "pair_target"):
                data = [(half[k]["x"][half[k]["on"]], fit_data[rec][k][arm]) for k in donors] if not replay else None
                identity = dict(registration=key, held=rec, fit=list(donors), seed=seed, arm=arm)
                name = f"{rec}_{seed}_{arm}"
                nets[arm], heads[name], n = trainer.fit(identity, name, seed, lambda: prior.model.network(width),
                    STEPS, None if replay else prior.sampler(data, False),
                    lambda net, x, y: model.loss(net, x, y, INC, arm == "pair_target"), replay, beat)
                updates += n
            z = {k: score(rec, half[rec] if k == "half" else prior.task(d, seed, f), d, nets, cuts)
                 for k, f in study.FRACTIONS.items()}
            old = next(r for r in ref["rows"][rec] if r["seed"] == seed)["scores"]
            for k, v in z.items():
                if {f: x for f, x in v["metrics"]["hindcast"].items() if f != "changed_from_hindcast"} != old[k]["metrics"]["hindcast"]:
                    raise ValueError("Changed frozen hindcast")
            rows.setdefault(rec, []).append(dict(seed=seed, fit=donors, scores=z))
            beat(phase="held", recording=rec, seed=seed, updates=updates)
    freeze = OUT/"head_freeze.json"
    run.once(freeze, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(freeze), rows=rows,
        summary=aggregate(rows), original_2pct=True, confirmation_access=False,
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
