"""Replay frozen TUMDOT decisions and solve strict three-reference capacity."""

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
from src.evaluation import m3w_tumdot_three_reference as model

run, torch, study, trainer, INC = parent.run, parent.torch, parent.study, parent.trainer, parent.INC
OUT = parent.OUT.parent/"tumdot_three_reference_v1"
PRIVATE = parent.PRIVATE.parent/OUT.name
ARMS = ("pair_guard", "pair_target", "cv_target")


def register():
    paths = [parent.OUT/f"{k}.json" for k in ("registration", "analysis", "head_freeze")]
    a, h = (run.read(paths[i]) for i in (1, 2))
    if (a["registration"] != run.sha(paths[0]) or a["heads"] != run.sha(paths[2])
            or a["registration"] != h["registration"] or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require frozen TUMDOT paired decision results")
    r = OUT/"registration.json"
    run.once(r, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(p) for p in paths], no_new_training=True, original_2pct=True, confirmation_access=False,
        risk="original all/easy positive harm and signed easy; fixed support and radius",
        primary="three-reference maximin over all existing same-radius directions with future labels",
        decomposition="complete, partial, absent row terms with unchanged full-roster CV denominator",
        label_aware_only=True, seconds=3, positive_pp=model.POSITIVE_PP, private_teacher_only=True)))
    return run.sha(r), a, h["heads"]


@torch.inference_mode()
def record(rec, q, d, nets, cuts, expected):
    cv, ols, bank, on, radius = (q[k] for k in ("cv", "ols", "bank", "on", "radius"))
    y, valid = d["future"], d["valid"]
    x = torch.from_numpy(q["x"].copy())
    x[..., -3:] = 0.
    scores = {k: parent.model.centered(v, x, INC).numpy() for k, v in nets.items()}
    use = {k: parent.model.choices(v, INC) for k, v in scores.items()}
    use["pair_guard"] = parent.model.choices(scores["pair_target"], INC, True)
    hindcast = bank[:, INC]
    observed = {}
    for k in ARMS:
        p = bank[np.arange(len(bank)), use[k]]
        np.testing.assert_array_equal((p != cv).any((1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1).max(1), radius, rtol=1e-7, atol=1e-13)
        z = model.mask_parts(hindcast, p, cv, y, valid)
        for f in ("lower", "observed"):
            np.testing.assert_allclose(sum(v[f] for v in z.values()), expected["contrasts"][k+"_vs_hindcast"][f],
                                       rtol=0., atol=1e-12)
        if sum(v["changed"] for v in z.values()) != expected["metrics"][k]["changed_from_hindcast"]:
            raise ValueError("Changed frozen decisions")
        observed[k] = z
    c = model.coefficients(cv, ols, bank, INC, on, y, valid, cuts[0])
    choice, sol = model.solve(c)
    if choice is not None:
        p = cv.copy()
        p[on] = bank[on, choice]
        np.testing.assert_array_equal((p != cv).any((1, 2)), on)
        np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1).max(1), radius, rtol=1e-7, atol=1e-13)
        v = study.source.score(p, cv, ols, y, valid, rec, cuts, hindcast)
        values = list(v["lower_pp"].values())
        analytic = c["offsets"]+c["gain"][:, np.arange(len(choice))*c["arms"]+choice].sum(1)
        np.testing.assert_allclose(values, 100*analytic/c["mass"], rtol=1e-9, atol=1e-10)
        exact = min(values)
        if not v["risk"]["supported"] or exact+1e-6 < sol["objective_pp"]:
            sol.update(status="postcheck_unresolved", passes=False)
            choice = None
        else:
            sol.update(passes=exact > model.POSITIVE_PP, lower=v["lower_pp"], min_pp=exact)
    sol.setdefault("passes", False)
    return dict(decomposition=observed, oracle=sol), choice


def aggregate(rows):
    result = {}
    sites = sorted({r.split("-")[0] for r in rows})
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    for part in study.FRACTIONS:
        by_site = {s: [x["scores"][part] for k, values in rows.items() if k.startswith(s+"-") for x in values]
                   for s in sites}
        flat = [x for v in by_site.values() for x in v]
        if len(flat) != 27:
            raise ValueError("Incomplete cohort")
        risk = [x["oracle"] for x in flat]
        oracle = dict(views=27, joint=sum(x["passes"] for x in risk),
            statuses={k: sum(x["status"] == k for x in risk) for k in sorted({x["status"] for x in risk})})
        if all("min_pp" in x for x in risk):
            oracle["mean_min_pp"] = float(np.mean([x["min_pp"] for x in risk]))
        decomp = {}
        for a in ARMS:
            decomp[a] = {}
            for k in model.SUBSETS:
                metrics = [z["decomposition"][a][k] for z in flat]
                out = {f: sum(v[f] for v in metrics) for f in ("rows", "changed")}
                for f in ("observed", "debit", "lower"):
                    v = np.array([np.mean([z["decomposition"][a][k][f] for z in by_site[s]]) for s in sites])
                    out[f] = dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist(),
                                  by_intersection=dict(zip(sites, v.tolist(), strict=True)))
                decomp[a][k] = out
        result[part] = dict(oracle=oracle, decomposition=decomp)
    return result


def execute(replay, beat):
    key, previous, weights = register()
    key0, lineage, base, _ = parent.prior.register()
    cache = parent.prior.prepare(key0, lineage, base, beat)
    cuts = np.asarray(lineage[-1]["open_dd_positive_complete_cv_cuts"])
    rows = {}
    for seed in study.parent.job.SEEDS:
        for rec, d in cache.items():
            file = PRIVATE/f"{rec}_{seed}.json"
            if file.exists() and not replay:
                s = run.read(file)
                if s["registration"] != key:
                    raise ValueError("Changed private receipt")
            else:
                cells = {k: parent.prior.task(d, seed, f) for k, f in study.FRACTIONS.items()}
                width = cells["half"]["x"].shape[-1]
                nets = {k: trainer.load(weights[f"{rec}_{seed}_{k}"], lambda: parent.prior.model.network(width),
                                        parent.STEPS) for k in ("cv_target", "pair_target")}
                ref = next(r for r in previous["rows"][rec] if r["seed"] == seed)["scores"]
                a, b = {}, {}
                for k, q in cells.items():
                    z, choice = record(rec, q, d, nets, cuts, ref[k])
                    a[k] = z
                    b[k] = choice.tolist() if choice is not None and z["oracle"]["passes"] else None
                s = dict(registration=key, seed=seed, scores=a, private_teacher=b)
                run.once(file, run.stable(s))
            rows.setdefault(rec, []).append(dict(seed=seed, scores=s["scores"]))
            beat(phase="held", recording=rec, seed=seed, complete_cells=sum(map(len, rows.values())))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, rows=rows, summary=aggregate(rows),
        original_2pct=True, confirmation_access=False, source_is_previously_exposed=True, no_new_training=True)))


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
            if p.get("complete_cells", 1) % 9 == 0 or p.get("complete"):
                print(z, flush=True)
        execute(a.replay, beat)
        beat(complete=True, replay=a.replay)


if __name__ == "__main__":
    main()
