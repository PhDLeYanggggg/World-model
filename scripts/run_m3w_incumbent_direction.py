"""Test direct incumbent-relative direction gains with fixed query recipients."""

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
from scripts import audit_m3w_moving_observability_cost as attribution
from src.world_model import m3w_incumbent_direction as model

study, run, h, torch = attribution.study, attribution.run, attribution.h, attribution.torch
readout = attribution.readout
OUT = study.OUT.parent / "incumbent_direction_v2"
PRIVATE = study.PRIVATE.parent / OUT.name
ARMS = ("incumbent", "direct", "absolute_matched", "no_debit_matched", "oracle")
STEPS = 500


@torch.inference_mode()
def menu(m, ix, groups, frozen):
    x = m.x[ix]
    n, k, width = x.shape
    raw = torch.from_numpy(x.reshape(n*k, width).astype(np.float32))
    z = frozen(raw).numpy().reshape(n, k, 3) * m.scale[ix, None, None]
    radius = np.linalg.norm(m.bank[ix]-m.floor[ix, None], axis=-1).max(2)
    initial = np.full(n, -1, dtype=int)
    for q in groups:
        initial[q] = study.choose(z[q], radius[q] > 0., study.CAP, False)
    take = np.flatnonzero(initial >= 0)
    if not len(take):
        return initial, take, [], np.zeros((0, k), bool), np.zeros((0, k), np.float32)
    compact = np.cumsum(initial >= 0)-1
    reduced = [compact[q[initial[q] >= 0]] for q in groups]
    left = m.bank[ix[take], initial[take]]
    pair_radius = np.linalg.norm(m.bank[ix[take]]-left[:, None], axis=-1).max(2)
    return initial, take, reduced, (radius[take] > 0.) & (pair_radius > 0.), pair_radius


def pair_arrays(m, ix, take, initial, y, valid, legal):
    rows = ix[take]
    x = m.x[rows]
    left = x[np.arange(len(take)), initial[take]]
    error = np.column_stack([readout.errors(p, y[rows], valid[rows])[0]
                             for p in m.bank[rows].swapaxes(0, 1)])
    gain = error[np.arange(len(take)), initial[take], None]-error
    return left, x, gain, legal & valid[rows].any(1)[:, None]


def batcher(m, records, frozen, fit, y, valid):
    banks = {s: [] for s in fit}
    for ix, groups in records.values():
        site = str(m.d["sites"][ix[0]])
        if site not in banks:
            continue
        a, take, _, legal, _ = menu(m, ix, groups, frozen)
        if not legal.any():
            continue
        left, right, gain, keep = pair_arrays(m, ix, take, a, y, valid, legal)
        j, k = np.nonzero(keep)
        if len(j):
            target = gain[j, k]/m.scale[ix[take[j]]].clip(1e-8)
            banks[site].append(tuple(np.asarray(v, np.float32) for v in (left[j], right[j, k], target)))
    active = [s for s, v in banks.items() if v]
    donors = [banks[s] for s in active]
    if not donors:
        raise ValueError("No fitting source has any known paired direction labels")

    def batch(rng):
        counts = rng.multinomial(128, np.full(len(donors), 1./len(donors)))
        out = []
        for recordings, count in zip(donors, counts, strict=True):
            for j in rng.integers(len(recordings), size=count):
                a, b, g = recordings[j]
                i = rng.integers(len(g))
                out.append((a[i], b[i], g[i]))
        return tuple(torch.from_numpy(np.stack(v)) for v in zip(*out, strict=True))
    return batch, active


@torch.inference_mode()
def score(m, rec, ix, groups, frozen, factor, net, context, y, valid, cuts):
    initial, take, reduced, legal, radius = menu(m, ix, groups, frozen)
    choice = {k: initial.copy() for k in ARMS}
    if legal.any():
        rows = ix[take]
        left, right, true_gain, _ = pair_arrays(m, ix, take, initial, y, valid, legal)
        n, k, width = right.shape
        c = np.broadcast_to(context[rows, None], (n, k, context.shape[1]))
        aux = np.concatenate((right, c), -1).reshape(n*k, -1).astype(np.float32)
        cond, _, missing = factor(torch.from_numpy(aux))
        cond = cond.numpy().reshape(n, k, 3)[..., 0]*m.scale[rows, None]
        p = missing.sigmoid().numpy().reshape(n, k)[:, :1]
        predicted = net(torch.from_numpy(np.broadcast_to(left[:, None], right.shape).copy().astype(np.float32)),
                        torch.from_numpy(right.astype(np.float32))).numpy()*m.scale[rows, None]
        lower = model.lower_gain(predicted, p, radius)
        observed_only = cond-cond[np.arange(n), initial[take], None]
        absolute = model.lower_gain(observed_only, p, radius)
        hindsight = np.where(valid[rows].any(1)[:, None], true_gain, -radius)
        choice["direct"][take], switches = model.single_swaps(lower, legal, reduced, initial[take])
        for name, scores, want in (("absolute_matched", absolute, switches),
                                    ("no_debit_matched", predicted, switches), ("oracle", hindsight, None)):
            choice[name][take], _ = model.single_swaps(scores, legal, reduced, initial[take], want)
    if any(not np.array_equal(initial >= 0, a >= 0) for a in choice.values()):
        raise ValueError("Edited recipients must remain unchanged")
    c, b = m.floor[ix], m.bank[ix]
    paths = {run.parent.CV: c, "ols": b[:, run.NAMES.index(run.parent.OLS)],
             **{a: run.model.selected_paths(v, b, c) for a, v in choice.items()}}
    pairs = {f"{a}_{r}": (r if r == "ols" else run.parent.CV, a, run.parent.CV)
             for a in ARMS for r in ("ols", "cv")}
    pairs.update({a: ("incumbent", a, run.parent.CV) for a in ARMS[1:]})
    pairs.update({f"direct_vs_{a}": (a, "direct", run.parent.CV) for a in ARMS[2:4]})
    q = m.d["queries"][ix]
    z = readout.score_recording(paths, pairs, y[ix], valid[ix], q, rec,
                                easy_cut=cuts["easy"], hard_cut=cuts["hard"])
    metrics = {}
    for a in ARMS:
        risk = z["metrics"][a]["risk"]
        dual = risk["supported"] and all(readout.merge_comparisons([z["comparisons"][f"{a}_{r}"]])
                                        ["completion_lower_pp"] > 0. for r in ("cv", "ols"))
        metrics[a] = dict(views=1, edits=risk["selected"], swaps=int((choice[a] != initial).sum()),
                          risk_pass=int(risk["supported"]), dual_positive=int(dual),
                          undefined=int(risk["completion_easy_risk_upper"] is None),
                          unknown_edits=risk["unknown_selected"])
    return metrics, z["comparisons"]


def execute(replay, beat):
    key, _, weights = study.register()
    f, a = (study.OUT / f"{k}.json" for k in ("head_freeze", "analysis"))
    attr = attribution.OUT / "analysis.json"
    p, audit = run.read(a), run.read(attr)
    if (p["registration"] != key or p["heads"] != run.sha(f) or audit["confirmation_access"]
            or audit["registration"] != run.sha(attribution.OUT / "registration.json")
            or not audit["original_2pct"]):
        raise ValueError("Require frozen observability comparison and attribution")
    freeze = run.read(f)["heads"]
    reg = OUT / "registration.json"
    trainer = study.old.study.trainer
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(trainer.__file__)), parents=[run.artifact(x) for x in (f, a, attr)],
        steps=STEPS, threads=4, workers=0, original_2pct=True, confirmation_access=False,
        target="signed paired known ADE relative to the frozen incumbent path, with an antisymmetric head",
        inference="frozen geometry missingness times exact incumbent-to-candidate path radius; max one swap per query",
        control="separate absolute-cost subtraction and no missingness debit match every realized swap query",
        fit="same seven donors, six reserved and one held; sample donors with known pairs, record empty donors",
        oracle="outcome-aware positive paired bound; diagnostic only", bootstrap=dict(draws=3000, seed=73107))))
    train_key = run.sha(reg)
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    context = study.model.observability_features(d["history"], d["groups"])
    cuts = run.cuts_by_source()
    parents, controls, ranks = h.cost.banks()
    trainer.PRIVATE = PRIVATE
    rows, heads, updates = {s: [] for s in p["by_site"]}, {}, 0
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        width = m.x.shape[-1]
        for ctrl in controls[m.seed]:
            role = ctrl["identity"]
            s, fit, cal = role["held"], role["fit"], role["cal_a"]+role["cal_b"]
            if (len(fit), len(cal), len(set(fit+cal+[s]))) != (7, 6, 14) or set(fit+cal+[s]) != set(rows):
                raise ValueError("Changed fitting roles")
            name = f"{m.seed}_{s}"
            frozen = trainer.load(weights[name], lambda: study.old.model.network(width), study.old.STEPS)
            factor = trainer.load(freeze[name], lambda: study.model.Head(width), study.STEPS)
            batch, active = batcher(m, records, frozen, fit, y, valid)
            ident = dict(registration=train_key, held=s, fit=fit, active=active, reserved=cal, seed=m.seed)
            net, heads[name], n = trainer.fit(ident, name, m.seed, lambda: model.PairedGain(width),
                                               STEPS, batch, model.loss, replay, beat)
            updates += n
            z = [score(m, rec, ix, groups, frozen, factor, net, context, y, valid, cuts[s])
                 for rec, (ix, groups) in records.items() if str(d["sites"][ix[0]]) == s]
            metrics = {a: {k: sum(r[0][a][k] for r in z) for k in z[0][0][a]} for a in ARMS}
            ref = next(r["metrics"]["entangled"] for r in p["by_site"][s] if r["seed"] == m.seed)
            if any(metrics["incumbent"][k] != v for k, v in ref.items() if k != "strong_dual"):
                raise ValueError("Frozen incumbent results changed")
            effects = {}
            for k in z[0][1]:
                x = readout.merge_comparisons([r[1][k] for r in z])
                effects[k] = dict(lower=x["completion_lower_pp"], observed=x["known_pp"],
                                  debit=x["known_pp"]-x["completion_lower_pp"])
            rows[s].append(dict(seed=m.seed, metrics=metrics, effects=effects, active_donors=len(active)))
            beat(phase="held", seed=m.seed, source=s, updates=updates)
        del m
    flat = [r for rs in rows.values() for r in rs]
    metrics = {a: {k: sum(r["metrics"][a][k] for r in flat) for k in flat[0]["metrics"][a]} for a in ARMS}
    if len(flat) != 42 or metrics["direct"]["views"] != 102:
        raise ValueError("Incomplete cohort")
    draw = np.random.default_rng(73107).integers(len(rows), size=(3000, len(rows)))

    def ci(k, f0):
        x = np.array([np.mean([r["effects"][k][f0] for r in rs]) for rs in rows.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    freeze_path = OUT / "head_freeze.json"
    run.once(freeze_path, run.stable(dict(registration=train_key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=train_key, heads=run.sha(freeze_path),
        by_site=rows, summary=dict(metrics=metrics, effects={k: {f0: ci(k, f0) for f0 in
          ("observed", "debit", "lower")} for k in flat[0]["effects"]}),
        original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**p):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **p)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", z)
            if p.get("phase") != "fit":
                print(z, flush=True)
        updates = execute(args.replay, beat)
        beat(complete=True, replay=args.replay, updates=updates)


if __name__ == "__main__":
    main()
