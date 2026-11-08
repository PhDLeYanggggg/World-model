"""Test explicit missingness estimation at frozen pedestrian query counts."""

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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
from scripts import run_m3w_moving_budget_risk as old
from src.world_model import m3w_moving_observability as model

run, h = old.run, old.h
OUT = old.OUT.parent/"observability_v1"
PRIVATE = old.PRIVATE.parent/OUT.name
ARMS = ("entangled", "gain_rank", "motion", "geometry", "no_missing", "geometry_free")
CAP = 4
STEPS = 500


def register():
    paths = [old.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "attribution", "attribution_registration")]
    a, h0, attr, reg = map(run.read, paths)
    key = old.register()
    if (a["registration"] != key or h0["registration"] != key or a["heads"] != run.sha(paths[1])
            or attr["registration"] != run.sha(paths[3]) or reg["parents"][0]["sha256"] != run.sha(paths[0])
            or a["confirmation_access"] or not a["original_2pct"]):
        raise ValueError("Require the frozen budget risk study and verified debit attribution")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(old.study.trainer.__file__)), parents=[run.artifact(p) for p in paths],
        steps=STEPS, threads=4, workers=0, native_steps=[8, 12], seeds=run.kin.prior.SEEDS,
        primary="geometry-aware factorization versus frozen entangled cap-4 at every identical query count",
        variants="motion-only and geometry missingness share conditional costs; no-missing ablation; free geometry diagnostic",
        target="wholly unobserved future mask labels a shared row probability; cost regression only on known outcomes",
        composition="conditional costs weighted by predicted observability; add exact radius times missing probability",
        ordering="same safe-first gain ranking; for count matches fill safe positions before other legal actions",
        fit="unchanged seven fitting, six reserved and one held development source; no held statistics",
        original_2pct=True, confirmation_access=False, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=73107, unit="14 source groups after averaging seeds"))))
    return run.sha(f), a, h0["heads"]


def batcher(m, context, target, legal, fit):
    donors = []
    for s in fit:
        rs = []
        for ix, groups in m.groups[s]:
            data = []
            for q in groups:
                i, k = np.nonzero(legal[ix[q]])
                if len(i):
                    data.append((ix[q[i]], k))
            if data:
                rs.append(data)
        if not rs:
            raise ValueError("Empty fitting source")
        donors.append(rs)

    def batch(rng):
        sizes = rng.multinomial(128, np.full(len(donors), 1./len(donors)))
        rows, arms = [], []
        for s, n in zip(donors, sizes, strict=True):
            for j in rng.integers(len(s), size=n):
                qs = s[j]
                i, k = qs[rng.integers(len(qs))]
                a = rng.integers(len(i))
                rows.append(i[a])
                arms.append(k[a])
        x = np.c_[m.x[rows, arms], context[rows]].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(target[rows, arms])
    return batch


def choose(z, legal, n, fill):
    gain = z[..., 0]
    safe = legal & (gain > 0.) & (z[..., 1:] <= 0.).all(2)
    a = old.model.choose(gain, safe, n)
    gap = n-int((a >= 0).sum())
    if fill and gap:
        b = old.model.choose(gain, legal & (a[:, None] < 0), gap)
        a[b >= 0] = b[b >= 0]
    return a


@torch.inference_mode()
def score(m, rec, ix, groups, y, valid, legal, context, net, frozen, cuts):
    c, b, x, scale = m.floor[ix], m.bank[ix], m.x[ix], m.scale[ix]
    target, mask, q = y[ix], valid[ix], m.d["queries"][ix]
    s = str(m.d["sites"][ix[0]])
    avail = legal[ix]
    n, k = avail.shape
    raw = torch.from_numpy(x.reshape(-1, x.shape[-1]).astype(np.float32))
    ent = frozen(raw).numpy().reshape(n, k, 3)*scale[:, None, None]
    ctx = np.broadcast_to(context[ix, None], (n, k, context.shape[1]))
    u, pm, pg = net(torch.from_numpy(np.concatenate((x, ctx), axis=-1).reshape(n*k, -1).astype(np.float32)))
    conditional = u.numpy().reshape(n, k, 3)*scale[:, None, None]
    mp, gp = (z.sigmoid().numpy().reshape(n, k)[:, :1] for z in (pm, pg))
    radius = np.linalg.norm(b-c[:, None], axis=-1).max(2)
    costs = dict(motion=model.costs(conditional, mp, radius),
                 geometry=model.costs(conditional, gp, radius), no_missing=conditional)
    gain = run.model.decision_costs(x, m.co[s], b, c, scale)[0]
    choices = {a: np.full(n, -1, dtype=int) for a in ARMS}
    for q0 in groups:
        a = choose(ent[q0], avail[q0], CAP, False)
        count = int((a >= 0).sum())
        choices["entangled"][q0] = a
        choices["gain_rank"][q0] = old.model.choose(gain[q0], avail[q0], count)
        for key, z in costs.items():
            choices[key][q0] = choose(z[q0], avail[q0], count, True)
        choices["geometry_free"][q0] = choose(costs["geometry"][q0], avail[q0], CAP, False)
        if any(int((choices[key][q0] >= 0).sum()) != count for key in ARMS[:-1]):
            raise ValueError("Matched query count changed")
    ols = b[:, run.NAMES.index(run.parent.OLS)]
    paths = {run.parent.CV: c, "ols": ols, **{a: run.model.selected_paths(v, b, c) for a, v in choices.items()}}
    pairs = {a+"_cv": (run.parent.CV, a, run.parent.CV) for a in ARMS}
    pairs.update({a+"_ols": ("ols", a, run.parent.CV) for a in ARMS})
    pairs.update({a+"_vs_entangled": ("entangled", a, run.parent.CV) for a in ARMS[1:]})
    pairs.update({f"geometry_vs_{a}": (a, "geometry", run.parent.CV) for a in ("motion", "no_missing", "gain_rank")})
    readout = run.parent.readout
    z = readout.score_recording(paths, pairs, target, mask, q, rec,
                                easy_cut=cuts[s]["easy"], hard_cut=cuts[s]["hard"])
    miss = ~mask.any(1)
    w = 1./m.d["crowd"][ix]
    comp = readout.comparison_masses(ols, c, c, target, mask, w)
    strong = comp["known_delta"]+comp["unknown_radius"] < 0.
    metrics = {}
    for a in ARMS:
        risk = z["metrics"][a]["risk"]
        both = risk["supported"] and all(readout.merge_comparisons([z["comparisons"][a+"_"+r]])["completion_lower_pp"] > 0.
                                        for r in ("cv", "ols"))
        metrics[a] = dict(views=1, edits=risk["selected"], risk_pass=int(risk["supported"]),
                          dual_positive=int(both), strong_dual=int(both and strong),
                          unknown_edits=risk["unknown_selected"],
                          undefined=int(risk["completion_easy_risk_upper"] is None))
    brier = {a: float(w@np.square(p[:, 0]-miss)/w.sum()) for a, p in (("motion", mp), ("geometry", gp))}
    return dict(metrics=metrics, comparisons=z["comparisons"], brier=brier)


def execute(key, previous, weights, replay, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    cuts = run.cuts_by_source()
    context = model.observability_features(d["history"], d["groups"])
    parents, controls, ranks = h.cost.banks()
    trainer = old.study.trainer
    trainer.PRIVATE = PRIVATE
    sites = sorted(set(d["sites"]))
    result, freeze, updates = {s: [] for s in sites}, {}, 0
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        terms, legal = old.cost_targets(m, y, valid, cuts)
        miss = np.broadcast_to((~valid.any(1))[:, None, None], (*terms.shape[:2], 1))
        target = np.concatenate((terms, miss), axis=2)
        for ctrl in controls[m.seed]:
            role = ctrl["identity"]
            s, fit, cal = role["held"], role["fit"], role["cal_a"]+role["cal_b"]
            if (len(fit), len(cal), len(set(fit+cal+[s]))) != (7, 6, 14) or set(fit+cal+[s]) != set(sites):
                raise ValueError("Preserve frozen roles")
            name = f"{m.seed}_{s}"
            ident = dict(registration=key, held=s, fit=fit, reserved=cal, seed=m.seed)
            n = m.x.shape[-1]
            frozen = trainer.load(weights[name], lambda: old.model.network(n), old.STEPS)
            net, freeze[name], u = trainer.fit(ident, name, m.seed, lambda: model.Head(n), STEPS,
                batcher(m, context, target, legal, fit), model.loss, replay, beat)
            updates += u
            rows = [score(m, rec, ix, groups, y, valid, legal, context, net, frozen, cuts)
                    for rec, (ix, groups) in records.items() if str(d["sites"][ix[0]]) == s]
            metrics = {a: {k: sum(r["metrics"][a][k] for r in rows) for k in rows[0]["metrics"][a]} for a in ARMS}
            ref = next(r for r in previous["by_site"][s] if r["seed"] == m.seed)
            if any(metrics["entangled"][k] != v for k, v in ref["metrics"]["screen_4"].items()):
                raise ValueError("Frozen cap-4 entangled policy did not replay")
            effects = {k: run.parent.readout.merge_comparisons([r["comparisons"][k] for r in rows])["completion_lower_pp"]
                       for k in rows[0]["comparisons"]}
            brier = {a: float(np.mean([r["brier"][a] for r in rows])) for a in ("motion", "geometry")}
            brier["geometry_minus_motion"] = brier["geometry"]-brier["motion"]
            result[s].append(dict(seed=m.seed, metrics=metrics, effects=effects, brier=brier))
            beat(phase="held", source=s, seed=m.seed, updates=updates)
        del m
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))

    def ci(f, k):
        x = np.array([np.mean([r[f][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    m = {k: {f: sum(r["metrics"][k][f] for r in flat) for f in flat[0]["metrics"][k]} for k in ARMS}
    if len(flat) != 42 or m["entangled"]["views"] != 102:
        raise ValueError("Require full pedestrian cohort")
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=dict(metrics=m, effects={k: ci("effects", k) for k in flat[0]["effects"]},
                     brier={k: ci("brier", k) for k in flat[0]["brier"]}),
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
    start = time.monotonic()
    with (PRIVATE/"process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**p):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **p)
            run.kin.prior.source.atomic(PRIVATE/"heartbeat.json", z)
            if p.get("phase") != "fit":
                print(z, flush=True)

        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
