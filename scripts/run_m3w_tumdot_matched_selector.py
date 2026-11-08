"""Fit fixed-menu selectors on exposed TUMDOT with site and day exclusion."""

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
from scripts import run_m3w_tumdot_matched_finite_transfer as study
from scripts import run_m3w_recording_count as trainer
from scripts.audit_m3w_opendd_finite_guard_capacity import best_risk
from src.world_model import m3w_tumdot_matched_selector as model

run, torch, source = study.run, study.torch, study.source
OUT = study.OUT.parent/"tumdot_matched_selector_v1"
PRIVATE = study.PRIVATE.parent/OUT.name
ARMS = ("hindcast", "probe", "utility", "no_probe", "risk", "oracle")
STEPS = 400


def register():
    key, *lineage = study.register()
    paths = [study.OUT/f"{k}.json" for k in ("registration", "analysis", "capacity_registration", "capacity")]
    a, cap = (run.read(paths[i]) for i in (1, 3))
    if (a["registration"] != key or cap["registration"] != run.sha(paths[2]) or a["confirmation_access"]
            or cap["confirmation_access"] or not a["source_is_previously_exposed"]):
        raise ValueError("Require frozen exposed TUMDOT results")
    reg = OUT/"registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in [*paths, Path(trainer.__file__)]],
        original_2pct=True, confirmation_access=False, steps=STEPS, threads=4, workers=0,
        fit="four donors excluding both held intersection and collection day; no held normalization or thresholds",
        sampling="uniform recordings then edited rows; all actions per sampled row",
        primary="learned utility versus fixed hindcast at original half radius",
        sensitivity="quarter radius scored on the same trained policy without refit",
        objective="per-step conservative CV gain and known all/easy positive harm, normalized by action radius",
        rank_rule="gain only; risk uses gain minus each nonnegative predicted harm; neither certifies the gate",
        roster="same exposed 9 recordings x 3 producer seeds, one campaign", seeds=study.parent.job.SEEDS,
        bootstrap=dict(draws=3000, seed=73107, unit="3 intersections after seed/day averaging; descriptive"))))
    return run.sha(reg), lineage, a, cap


def prepare(reg, lineage, previous, beat):
    fit, tangent, prior, warm, fitted, historical = lineage
    models = None
    registry = run.read(source.PUBLIC/"registration.json")["files"]
    cache = {}
    for rec, (name, _) in source.ROSTER.items():
        p, meta = PRIVATE/f"{rec}.npz", PRIVATE/f"{rec}_cache.json"
        if p.exists() and meta.exists():
            z = run.read(meta)
            if z["registration"] != reg or z["file"]["sha256"] != run.sha(p):
                raise ValueError("Changed feature cache")
        else:
            if models is None:
                models = study.frozen_models(fit, tangent, prior, warm, fitted)
            tracks, _, ids = source.read_tracks(source.SOURCE/name)
            h, y, starts = source.windows(tracks)
            valid = np.isfinite(y).all(2)
            old = historical["recorded_files"][rec]
            if (len(h), len(starts), int(valid.all(1).sum()), int((~valid.any(1)).sum())) != (
                    old["targets"], old["queries"], old["complete"], old["unknown"]):
                raise ValueError("Changed frozen TUMDOT roster")
            data = dict(history=h, future=y, valid=valid, ids=np.asarray(sorted(ids)))
            for seed in study.parent.job.SEEDS:
                cv, ols, base, _, deltas = study.forecast(h, models[seed])
                data.update({f"{seed}_{k}": v for k, v in dict(cv=cv, ols=ols, base=base, **deltas).items()})
            tmp = p.with_suffix(".pending.npz")
            np.savez(tmp, **data)
            tmp.replace(p)
            run.once(meta, run.stable(dict(registration=reg, file=run.artifact(p), source=registry[rec])))
        with np.load(p, allow_pickle=False) as z:
            cache[rec] = {k: z[k] for k in z.files}
        beat(phase="features", recording=rec, completed=len(cache))
    if sum(len(d["history"]) for d in cache.values()) != 1456 or len(previous["recordings"]) != len(cache):
        raise ValueError("Incomplete TUMDOT roster")
    return cache


def task(d, seed, fraction):
    cv, ols, base = (d[f"{seed}_{k}"] for k in ("cv", "ols", "base"))
    paths, on = study.matched_paths(cv, base, {k: d[f"{seed}_{k}"] for k in study.ARMS}, fraction)
    bank = np.stack([paths[k] for k in study.ARMS], axis=1)
    radius = np.linalg.norm(base-cv, axis=-1).max(1)*fraction
    x, probe = model.features(d["history"], cv, ols, bank, radius)
    return dict(cv=cv, ols=ols, bank=bank, radius=radius, on=on, x=x, probe=probe)


def sampler(cells, probes):
    def batch(rng):
        i = rng.integers(len(cells), size=128)
        indices = [rng.integers(len(x), size=int((i == k).sum())) for k, (x, _) in enumerate(cells)]
        x, y = [np.concatenate([v[j][i] for v, i in zip(cells, indices, strict=True)]) for j in (0, 1)]
        if not probes:
            x[..., -3:] = 0.
        return torch.from_numpy(x), torch.from_numpy(y)
    return batch


@torch.inference_mode()
def evaluate(rec, z, d, models, cuts):
    cv, ols, bank, radius, on = (z[k] for k in ("cv", "ols", "bank", "radius", "on"))
    y, valid = d["future"], d["valid"]
    x = torch.from_numpy(z["x"])
    score = models["full"](x).numpy()
    x0 = x.clone()
    x0[..., -3:] = 0.
    blind = models["no_probe"](x0).numpy()[..., 0]
    paths = dict(hindcast=bank[:, study.ARMS.index("hindcast")], probe=model.choose(bank, z["probe"]),
        utility=model.choose(bank, score[..., 0]), no_probe=model.choose(bank, blind),
        risk=model.choose(bank, score[..., 0]-score[..., 1:].clip(0.).sum(2)),
        oracle=best_risk(cv, list(bank.swapaxes(0, 1)), y, valid))
    metrics, contrasts = {}, {}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    for k, p in paths.items():
        np.testing.assert_array_equal(np.any(p != cv, axis=(1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1).max(1), radius, atol=1e-13, rtol=1e-7)
        metrics[k] = source.score(p, cv, ols, y, valid, rec, cuts, paths["hindcast"])
        metrics[k]["edits"] = int(on.sum())
        known, lower = study.paired(paths["hindcast"], p, y, valid)
        contrasts[k+"_vs_hindcast"] = dict(lower=float(100*lower.sum()/mass), observed=float(100*known.sum()/mass))
    for a, b in (("utility", "no_probe"), ("utility", "probe"), ("risk", "utility")):
        known, lower = study.paired(paths[b], paths[a], y, valid)
        contrasts[a+"_vs_"+b] = dict(lower=float(100*lower.sum()/mass), observed=float(100*known.sum()/mass))
    return dict(metrics=metrics, contrasts=contrasts)


def execute(replay, beat):
    key, lineage, previous, capacity = register()
    cache = prepare(key, lineage, previous, beat)
    old = {r["recording"]: r for r in previous["recordings"]}
    witness = {(r["recording"], r["seed"]): r for r in capacity["rows"]}
    cuts = np.asarray(lineage[-1]["open_dd_positive_complete_cv_cuts"])
    trainer.PRIVATE = PRIVATE
    results, weights, updates = {}, {}, 0
    for seed in study.parent.job.SEEDS:
        half = {k: task(v, seed, .5) for k, v in cache.items()}
        for rec, d in cache.items():
            fit = model.fit_roster(rec, cache)
            ids = set(d["ids"])
            if len(fit) != 4 or any(not ids.isdisjoint(cache[r]["ids"]) for r in fit):
                raise ValueError("Held recording overlaps its fitting sources or track identifiers")
            data = []
            if not replay:
                for r in fit:
                    q, ref = half[r], cache[r]
                    on = q["on"]
                    label = model.targets(q["cv"][on], q["bank"][on], ref["future"][on], ref["valid"][on],
                                          q["radius"][on], cuts[0])
                    data.append((q["x"][on], label))
            nets = {}
            for a in ("full", "no_probe"):
                name = f"{rec}_{seed}_{a}"
                identity = dict(registration=key, held=rec, fit=list(fit), seed=seed, arm=a)
                width = half[rec]["x"].shape[-1]
                nets[a], weights[name], n = trainer.fit(identity, name, seed, lambda: model.network(width), STEPS,
                    None if replay else sampler(data, a == "full"), model.loss, replay, beat)
                updates += n
            by_radius = {}
            ref = next(r for r in old[rec]["seeds"] if r["seed"] == seed)
            for k, f in study.FRACTIONS.items():
                q = half[rec] if k == "half" else task(d, seed, f)
                z = evaluate(rec, q, d, nets, cuts)
                z0 = z["metrics"]["hindcast"]
                for n in ("risk", "edits", "available_ade", "available_fde"):
                    if z0[n] != ref[k]["policies"]["hindcast"][n]:
                        raise ValueError("Changed frozen hindcast score")
                oracle = witness[rec, seed][k]["capacity"]["expanded"]["risk_optimum"]
                np.testing.assert_allclose(z["metrics"]["oracle"]["lower_pp"]["cv"], oracle["lower"]["cv"],
                                           atol=1e-12, rtol=0.)
                if z["metrics"]["oracle"]["risk"]["supported"] != oracle["passed"]:
                    raise ValueError("Changed matched risk oracle")
                by_radius[k] = z
            results.setdefault(rec, []).append(dict(seed=seed, fit=fit, scores=by_radius))
            beat(phase="held", recording=rec, seed=seed, updates=updates)
    freeze = OUT/"head_freeze.json"
    run.once(freeze, run.stable(dict(registration=key, heads=weights)))
    summary = aggregate(results)
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(freeze), rows=results,
        summary=summary, source_is_previously_exposed=True, campaigns=1, confirmation_access=False,
        original_2pct=True, no_deployment=True)))
    return updates


def aggregate(results):
    groups = sorted({k.split("-")[0] for k in results})
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    out = {}
    for part in study.FRACTIONS:
        rows = {k: [r["scores"][part] for r in v] for k, v in results.items()}
        flat = [r for v in rows.values() for r in v]
        if len(flat) != 27:
            raise ValueError("Missing registered views")

        def ci(fn):
            v = np.asarray([np.mean([fn(r) for k, z in rows.items() if k.startswith(g+"-") for r in z])
                            for g in groups])
            return dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist(),
                        by_intersection=dict(zip(groups, v.tolist(), strict=True)))

        metrics = {}
        for a in ARMS:
            r = [z["metrics"][a] for z in flat]
            metrics[a] = dict(risk_pass=sum(m["risk"]["supported"] for m in r),
                joint_cv=sum(m["risk"]["supported"] and m["lower_pp"]["cv"] > 0. for m in r),
                joint_cv_ols=sum(m["risk"]["supported"] and min(m["lower_pp"]["cv"], m["lower_pp"]["ols4"]) > 0. for m in r),
                easy_undefined=sum(m["risk"]["completion_easy_risk_upper"] is None for m in r),
                worst_easy_pct=max((100*m["risk"]["known_easy_net_degradation"] for m in r
                                    if m["risk"]["known_easy_net_degradation"] is not None), default=None),
                ade=float(np.mean([m["available_ade"] for m in r])),
                fde=float(np.mean([m["available_fde"] for m in r])),
                cv=ci(lambda z: z["metrics"][a]["lower_pp"]["cv"]),
                ols4=ci(lambda z: z["metrics"][a]["lower_pp"]["ols4"]))
        effects = {k: {f: ci(lambda z: z["contrasts"][k][f]) for f in ("lower", "observed")}
                   for k in flat[0]["contrasts"]}
        out[part] = dict(metrics=metrics, effects=effects, views=len(flat))
    return out


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
