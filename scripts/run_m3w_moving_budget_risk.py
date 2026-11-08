"""Test a learned cost screen at larger, count-matched query budgets."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
from scripts import audit_m3w_moving_capacity_factorial as fact
from src.world_model import m3w_moving_budget_risk as model

study, run, h = fact.study, fact.run, fact.h
OUT = fact.OUT.parent/"budget_risk_v1"
PRIVATE = study.PRIVATE.parent/OUT.name
CAPS = (1, 4, 8)
ARMS = tuple(f"{a}_{cap}" for cap in CAPS for a in ("screen", "gain", "blind"))
STEPS = 500


def register():
    p, r = (fact.OUT/f"{k}.json" for k in ("analysis", "registration"))
    prior = run.read(p)
    if prior["registration"] != run.sha(r) or prior["confirmation_access"] or not prior["original_2pct"]:
        raise ValueError("Require frozen budget-factorial audit")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(study.trainer.__file__)), parents=[run.artifact(p) for p in
            (p, r, study.OUT/"registration.json", study.OUT/"analysis.json")],
        caps=CAPS, primary="cap 4 screened vs original gain at same exact query counts",
        target="paired CV lower gain, all/easy positive-harm excess including missingness radius",
        screen="positive predicted gain and nonpositive predicted all/easy excess; no threshold fit",
        controls="same exact causal query count; frozen gain rank and risk-blind learned gain rank",
        donor_roles="original seven fitting, six reserved, one held; no reserved/held normalization",
        original_2pct=True, confirmation_access=False, steps=STEPS, workers=0, threads=4,
        native_steps=[8, 12], seeds=run.kin.prior.SEEDS, no_threshold_search=True,
        bootstrap=dict(draws=3000, seed=73107, unit="14 source groups after averaging seeds"))))
    return run.sha(f)


def cost_targets(m, y, valid, cuts):
    n = len(y)
    c = run.parent.readout.errors(m.floor, y, valid)[0]
    e = np.column_stack([run.parent.readout.errors(p, y, valid)[0] for p in m.bank.swapaxes(0, 1)])
    radius = np.linalg.norm(m.bank-m.floor[:, None], axis=-1).max(2)
    known = valid.any(1)
    cut = np.empty(n)
    for s, ix in m.ix.items():
        cut[ix] = cuts[s]["easy"]
    easy = known & (c > 0.) & (c <= cut)
    gain = np.where(known[:, None], c[:, None]-e, 0.)
    unknown = np.where(known[:, None], 0., radius)
    harm = (-gain).clip(min=0.)
    all_excess = harm+unknown-.02*np.where(known, c, 0.)[:, None]
    easy_excess = np.where(easy[:, None], harm-.02*c[:, None], 0.)+unknown
    target = np.stack((gain-unknown, all_excess, easy_excess), axis=2)/m.scale[:, None, None].clip(1e-8)
    return target.clip(-8., 8.).astype(np.float32), radius > 0.


def batcher(m, target, legal, fit):
    donors = []
    for s in fit:
        rs = []
        for ix, groups in m.groups[s]:
            data = []
            for q in groups:
                j, k = np.nonzero(legal[ix[q]])
                if len(j):
                    data.append((ix[q[j]], k))
            if data:
                rs.append(data)
        if not rs:
            raise ValueError("Empty fitting source")
        donors.append(rs)

    def batch(rng):
        sizes = rng.multinomial(128, np.full(len(donors), 1./len(donors)))
        rows, kinds = [], []
        for s, n in zip(donors, sizes, strict=True):
            for i in rng.integers(len(s), size=n):
                qs = s[i]
                a, b = qs[rng.integers(len(qs))]
                j = rng.integers(len(a))
                rows.append(a[j])
                kinds.append(b[j])
        return (torch.from_numpy(m.x[rows, kinds].astype(np.float32)),
                torch.from_numpy(target[rows, kinds]))
    return batch


@torch.inference_mode()
def score(m, rec, ix, groups, y, valid, legal, net, cuts):
    c, b, x = m.floor[ix], m.bank[ix], m.x[ix]
    target, mask, q = y[ix], valid[ix], m.d["queries"][ix]
    s = str(m.d["sites"][ix[0]])
    avail = legal[ix]
    z = net(torch.from_numpy(x.reshape(-1, x.shape[-1]).astype(np.float32))).numpy().reshape(*avail.shape, 3)
    raw = z[..., 0]*m.scale[ix, None]
    safe = avail & (raw > 0.) & (z[..., 1:] <= 0.).all(-1)
    old = run.model.decision_costs(x, m.co[s], b, c, m.scale[ix])[0]
    choices = {k: np.full(len(ix), -1, dtype=int) for k in ARMS}
    for q0 in groups:
        for cap in CAPS:
            a = model.choose(raw[q0], safe[q0], cap)
            n = int((a >= 0).sum())
            choices[f"screen_{cap}"][q0] = a
            for k, f in (("gain", old), ("blind", raw)):
                shadow = model.choose(f[q0], avail[q0], n)
                if int((shadow >= 0).sum()) != n:
                    raise ValueError("Matched query counts disagree")
                choices[f"{k}_{cap}"][q0] = shadow
    ols = b[:, run.NAMES.index(run.parent.OLS)]
    paths = {run.parent.CV: c, "ols": ols, **{a: run.model.selected_paths(z, b, c) for a, z in choices.items()}}
    pairs = {a+"_cv": (run.parent.CV, a, run.parent.CV) for a in ARMS}
    pairs.update({a+"_ols": ("ols", a, run.parent.CV) for a in ARMS})
    for cap in CAPS:
        for a in ("gain", "blind"):
            pairs[f"screen_{cap}_vs_{a}"] = (f"{a}_{cap}", f"screen_{cap}", run.parent.CV)
    for cap in (4, 8):
        pairs[f"screen_{cap}_vs_1"] = ("screen_1", f"screen_{cap}", run.parent.CV)
    readout = run.parent.readout
    res = readout.score_recording(paths, pairs, target, mask, q, rec,
                                 easy_cut=cuts[s]["easy"], hard_cut=cuts[s]["hard"])
    w = 1./m.d["crowd"][ix]
    comp = readout.comparison_masses(ols, c, c, target, mask, w)
    strong = comp["known_delta"]+comp["unknown_radius"] < 0.
    stats = {}
    for a in ARMS:
        risk = res["metrics"][a]["risk"]
        both = risk["supported"] and all(readout.merge_comparisons([res["comparisons"][a+"_"+k]])["completion_lower_pp"] > 0.
                                        for k in ("cv", "ols"))
        stats[a] = dict(views=1, edits=risk["selected"], risk_pass=int(risk["supported"]),
                        dual_positive=int(both), strong_dual=int(strong and both),
                        undefined=int(risk["completion_easy_risk_upper"] is None))
    return dict(metrics=stats, comparisons=res["comparisons"])


def execute(key, replay, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    cuts = run.cuts_by_source()
    parents, controls, ranks = h.cost.banks()
    study.trainer.PRIVATE = PRIVATE
    sites = sorted(set(d["sites"]))
    result, freeze, updates = {s: [] for s in sites}, {}, 0
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        target, legal = cost_targets(m, y, valid, cuts)
        for ctrl in controls[m.seed]:
            role = ctrl["identity"]
            s, fit, cal = role["held"], role["fit"], role["cal_a"]+role["cal_b"]
            if (len(fit), len(cal), len(set(fit+cal+[s]))) != (7, 6, 14) or set(fit+cal+[s]) != set(sites):
                raise ValueError("Preserve original donor splits")
            name = f"{m.seed}_{s}"
            ident = dict(registration=key, held=s, fit=fit, reserved=cal, seed=m.seed)
            net, freeze[name], n = study.trainer.fit(ident, name, m.seed, lambda: model.network(m.x.shape[-1]),
                STEPS, batcher(m, target, legal, fit), model.loss, replay, beat)
            updates += n
            scored = [score(m, rec, ix, groups, y, valid, legal, net, cuts) for rec, (ix, groups) in records.items()
                      if str(d["sites"][ix[0]]) == s]
            result[s].append(dict(seed=m.seed,
                metrics={k: {f: sum(r["metrics"][k][f] for r in scored) for f in scored[0]["metrics"][k]} for k in ARMS},
                effects={k: run.parent.readout.merge_comparisons([r["comparisons"][k] for r in scored])["completion_lower_pp"]
                         for k in scored[0]["comparisons"]}))
            beat(phase="held", source=s, seed=m.seed, updates=updates)
        del m
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    metrics = {k: {f: sum(r["metrics"][k][f] for r in flat) for f in flat[0]["metrics"][k]} for k in ARMS}
    if metrics["screen_4"]["views"] != 102 or len(flat) != 42:
        raise ValueError("Require original 102 views")
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=dict(metrics=metrics, effects={k: ci(k) for k in flat[0]["effects"]}),
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

        u = execute(register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
