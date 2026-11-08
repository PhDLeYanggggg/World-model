"""Compare first-offer slate recall and one-action decisions at a fixed radius."""

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
from scripts import run_m3w_radius_opening as parent
from src.world_model import m3w_opening_slate as model

run, h, torch, readout = parent.run, parent.h, parent.torch, parent.readout
OUT = parent.OUT.parent/"opening_slate_v2"
PRIVATE = parent.PRIVATE.parent/OUT.name
ARMS = ("incumbent", "frozen", "ridge", "pointwise", "listwise")
MODES = ("flat", "row", "direction")
KS = (1, 4, 8, 16)
STEPS = 300


@torch.inference_mode()
def offer(m, ix, groups, frozen, context):
    p = parent.prepare(m, ix, groups, frozen)
    if p["offers"]:
        r = p["offers"][0]
        row = r["row"]
        path = r["path"]
        motion = path-p["cv"][row]
        r["delta"] = motion[np.arange(len(row)), np.linalg.norm(motion, axis=-1).argmax(1)]
        scores = frozen(torch.from_numpy(r["x"].astype(np.float32))).numpy()
        r["raw"] = scores[:, 0]*m.scale[ix[row]]
        r["input"] = np.c_[r["x"], context[ix[row]], r["delta"]/r["radius"], scores].astype(np.float32)
    return p


def batcher(donors):
    sources = [r for r in donors.values() if r]
    if not sources:
        raise ValueError("No eligible fitting offer")

    def batch(rng):
        x, y = [], []
        for i in rng.integers(len(sources), size=8):
            source = sources[i]
            r = source[rng.integers(len(source))]
            k = rng.choice(len(r["input"]), size=64, replace=len(r["input"]) < 64)
            x.append(r["input"][k])
            y.append(r["quality"][k])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y).astype(np.float32))
    return batch


@torch.inference_mode()
def score(p, nets, ridge, target, valid, q, rec, cut):
    paths = {run.parent.CV: p["cv"], "ols": p["ols"], "incumbent": p["original"]}
    scores, indices, r = {}, {}, None
    if p["offers"]:
        r = p["offers"][0]
        scores = dict(frozen=r["raw"], ridge=r["x"]@ridge,
                      **{k: n(torch.from_numpy(r["input"])).squeeze(1).numpy() if n is not None else r["raw"]
                         for k, n in nets.items()})
    for a in ARMS[1:]:
        paths[a] = p["original"].astype(float)
        if r is not None:
            i = int(scores[a].argmax())
            if a == "frozen" and i != r["frozen"]:
                raise ValueError("Changed prior frozen rank")
            row = r["row"][i]
            paths[a][row] = r["path"][i]
            np.testing.assert_allclose(np.linalg.norm(paths[a][row]-p["cv"][row], axis=-1).max(), r["radius"],
                                       atol=1e-7, rtol=1e-6)
            indices[a] = i
    pairs = {f"{a}_{b}": (b, a, run.parent.CV) for a in ARMS for b in (run.parent.CV, "ols", "incumbent")}
    pairs.update({f"listwise_vs_{a}": (a, "listwise", run.parent.CV) for a in ARMS[1:-1]})
    z = readout.score_recording(paths, pairs, target, valid, q, rec, easy_cut=cut["easy"], hard_cut=cut["hard"])
    parent.add_labels(p, target, valid, cut)
    counts = {}
    for a in ARMS:
        risk = z["metrics"][a]["risk"]
        terms = [z["comparisons"][f"{a}_{b}"] for b in (run.parent.CV, "ols", "incumbent")]
        joint = risk["supported"] and all(t["known_delta"]-t["unknown_radius"] > 0. for t in terms)
        if a in indices and joint != r["joint"][indices[a]]:
            raise ValueError("Changed marginal action label")
        counts[a] = dict(views=1, edits=risk["selected"], opened=int(a in indices),
                         risk=int(risk["supported"]), joint=int(joint),
                         undefined=int(risk["completion_easy_risk_upper"] is None))
    recall = {"all": int(r is not None and r["joint"].any())}
    for a in ARMS[1:]:
        for mode in MODES:
            for k in KS:
                ix = model.shortlist(scores[a], r["row"], r["delta"], k, mode) if r is not None else []
                recall[f"{a}_{mode}_{k}"] = int(r is not None and r["joint"][ix].any())
    return counts, recall, z["comparisons"]


def execute(replay, beat):
    pa, pr = (parent.OUT/f"{k}.json" for k in ("analysis", "registration"))
    previous = run.read(pa)
    if previous["registration"] != run.sha(pr) or previous["confirmation_access"]:
        raise ValueError("Require frozen first-offer data roles")
    _, _, weights = parent.parent.parent.study.register()
    registration = OUT/"registration.json"
    trainer = parent.parent.parent.study.old.study.trainer
    ridge_files = {f'{r["seed"]}_{s}': run.artifact(parent.PRIVATE/f'{r["seed"]}_{s}.json')
                   for s, rows in previous["by_site"].items() for r in rows}
    aborted = OUT.parent/"opening_slate_v1"/"registration.json"
    run.once(registration, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        parents=[run.artifact(x) for x in (pa, pr, Path(trainer.__file__))],
        frozen_ridge_coefficients=ridge_files,
        aborted=run.artifact(aborted),
        original_2pct=True, confirmation_access=False, steps=STEPS, native_steps=[8, 12], threads=4, workers=0,
        fit="7 donors; reserved 6 untouched; 1 held; same draws and initialization for loss ablation",
        no_offer_donors="no model fit; both learned arms replay the frozen rank at matched budget",
        radius="exact first-offer radius and row eligibility from parent",
        recall="label-aware coverage only; agent or sector diversity at fixed shortlist size",
        intervention="one action for each first offer, exact count and radius per recording",
        primary="listwise versus pointwise lower utility at identical intervention counts and radii",
        no_held_tuning=True, bootstrap=dict(draws=3000, seed=73107))))
    key = run.sha(registration)
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    cuts = run.cuts_by_source()
    parents, controls, ranks = h.cost.banks()
    obs = parent.parent.parent.study.model
    context = obs.observability_features(d["history"], d["groups"])
    trainer.PRIVATE = PRIVATE
    rows = {s: [] for s in previous["by_site"]}
    head_freeze, updates = {}, 0
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        width = m.x.shape[-1]
        for c in controls[m.seed]:
            i = c["identity"]
            s, fit, cal = i["held"], i["fit"], i["cal_a"]+i["cal_b"]
            if (len(fit), len(cal), len(set(fit+cal+[s]))) != (7, 6, 14) or set(fit+cal+[s]) != set(rows):
                raise ValueError("Changed frozen source roles")
            name = f"{m.seed}_{s}"
            frozen = trainer.load(weights[name], lambda: parent.parent.parent.study.old.model.network(width),
                                  parent.parent.parent.study.old.STEPS)
            donor = {k: [] for k in fit}
            held = []
            for rec, (ix, groups) in records.items():
                src = str(d["sites"][ix[0]])
                if src == s:
                    held.append((rec, ix, groups))
                elif src in donor:
                    p = offer(m, ix, groups, frozen, context)
                    if p["offers"]:
                        donor[src].append(parent.add_labels(p, y[ix], valid[ix], cuts[src])[0])
            saved = run.read(run.checked(ridge_files[name]))
            if saved["registration"] != run.sha(pr):
                raise ValueError("Changed ridge parent")
            receipt = next(v for v in previous["by_site"][s] if v["seed"] == m.seed)
            for f in ("metrics", "oracle", "effects"):
                if saved[f] != receipt[f]:
                    raise ValueError("Changed frozen scalar")
            ridge = np.asarray(saved["coeff"])
            support = sum(bool(v) for v in donor.values())
            batch = None if replay or not support else batcher(donor)
            nets = {}
            for a in ("pointwise", "listwise"):
                identity = dict(registration=key, held=s, seed=m.seed, fit=fit, reserved=cal, arm=a)
                name_a = name+"_"+a
                if not support:
                    nets[a] = None
                    head_freeze[name_a] = dict(identity=identity, status="no_donor_offer_frozen_rank")
                    continue
                nets[a], head_freeze[name_a], u = trainer.fit(identity, name_a, m.seed,
                    lambda: model.network(width+context.shape[1]+5), STEPS, batch, getattr(model, a), replay, beat)
                updates += u
            z = [score(offer(m, ix, g, frozen, context), nets, ridge, y[ix], valid[ix], d["queries"][ix], rec, cuts[s])
                 for rec, ix, g in held]
            counts = {a: {f: sum(r[0][a][f] for r in z) for f in z[0][0][a]} for a in ARMS}
            if any(counts[a][f] != receipt["metrics"][b][f] for a, b in (("frozen", "frozen_first"),
                   ("ridge", "learned_first"), ("incumbent", "incumbent")) for f in counts[a]):
                raise ValueError("Fixed first-offer controls did not replay")
            recall = {k: sum(r[1][k] for r in z) for k in z[0][1]}
            if recall["all"] != receipt["oracle"]["first_joint"]:
                raise ValueError("Changed exact first-offer capacity")
            effects = {k: readout.merge_comparisons([r[2][k] for r in z])["completion_lower_pp"] for k in z[0][2]}
            for a, b in (("frozen", "frozen_first"), ("ridge", "learned_first")):
                np.testing.assert_allclose(effects[a+"_incumbent"], receipt["effects"][b+"_incumbent"],
                                           rtol=0., atol=1e-12)
            rows[s].append(dict(seed=m.seed, metrics=counts, recall=recall, effects=effects,
                                fit_sources_with_offers=support, fit_offers=sum(map(len, donor.values()))))
            beat(cells=sum(map(len, rows.values())), seed=m.seed, updates=updates)
        del m
    flat = [r for rs in rows.values() for r in rs]
    if len(flat) != 42:
        raise ValueError("Incomplete cohort")
    metrics = {k: {f: sum(r["metrics"][k][f] for r in flat) for f in flat[0]["metrics"][k]} for k in ARMS}
    recall = {k: sum(r["recall"][k] for r in flat) for k in flat[0]["recall"]}
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))
    effects = {}
    for k in flat[0]["effects"]:
        v = np.array([np.mean([r["effects"][k] for r in rs]) for rs in rows.values()])
        effects[k] = dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
    out = OUT/"head_freeze.json"
    run.once(out, run.stable(dict(registration=key, heads=head_freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(out), by_site=rows, metrics=metrics,
        recall=recall, effects=effects, original_2pct=True, confirmation_access=False)))
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
            if "cells" in p and p["cells"] % 14 == 0:
                print(z, flush=True)
        updates = execute(a.replay, beat)
        beat(cells=42, complete=True, replay=a.replay, updates=updates)


if __name__ == "__main__":
    main()
