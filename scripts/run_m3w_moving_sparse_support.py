"""Audit and learn sparse support on the existing pedestrian development bank."""

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
from scripts import run_m3w_record_rescue as parent
from scripts import run_m3w_recording_count as trainer
from src.world_model import m3w_moving_sparse_support as model

h, run = parent.h, parent.run
OUT = parent.PUBLIC.parent/"sparse_support_v1"
PRIVATE = parent.PRIVATE.parent/OUT.name
ARMS = ("cv", "ols", "direct", "gain", "cv_head", "dual_head",
        "first_cv_oracle", "first_dual_oracle", "window_cv_oracle", "window_dual_oracle")
LEARNED = ("cv_head", "dual_head")
STEPS = 500


def register():
    p = parent.PUBLIC/"analysis.json"
    if run.read(parent.PUBLIC/"replay_execution.json")["artifact"]["sha256"] != run.sha(p):
        raise ValueError("Require exact replay of the dense policy")
    parents = [p, parent.REG, h.cost.REG, h.cost.PUBLIC/"freeze.json", run.REG]
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(trainer.__file__)), parents=[run.artifact(p) for p in parents],
        sites=14, recordings=34, views=102, seeds=run.kin.prior.SEEDS, steps=STEPS, threads=4, workers=0,
        native_steps=[8, 12], action="one intervention from CV per recording at first legally actionable query",
        labels="joint pass against CV alone or CV and OLS4; exact query-weighted observed plus unknown debit",
        fit="original 7/6/1 roles; fit only 7 training sources; uniform source, recording, query, action",
        controls="old causal gain rank at same first query and count; dense direct and OLS4 contextual",
        oracles="keep same candidate bank, first query or all queries; fallback if no witness",
        original_2pct=True, no_threshold_search=True, confirmation_access=False,
        bootstrap=dict(draws=3000, seed=73107, unit="14 source groups after averaging seeds"))))
    return run.sha(f)


def rows_for_record(m, ix, y, valid, ols, cut):
    cv, bank = m.floor[ix], m.bank[ix]
    target, mask = y[ix], valid[ix]
    weights = 1./m.d["crowd"][ix]
    score = run.parent.readout
    errors = score.errors
    c = errors(cv, target, mask)[0]
    e = np.column_stack([errors(p, target, mask)[0] for p in bank.swapaxes(0, 1)])
    comp = score.comparison_masses(ols[ix], cv, cv, target, mask, weights)
    offset = comp["known_delta"]-comp["unknown_radius"]
    labels = model.targets(c, e, mask.any(1), cut, weights, offset)
    legal = (bank != cv[:, None]).any((2, 3))
    return labels & legal[:, :, None], legal, float(offset), c[:, None]-e


def prepare(m, records, y, valid):
    cuts = run.cuts_by_source()
    ols = m.bank[:, run.NAMES.index(run.parent.OLS)]
    result = {}
    for rec, (ix, groups) in records.items():
        s = str(m.d["sites"][ix[0]])
        labels, legal, offset, benefit = rows_for_record(m, ix, y, valid, ols, cuts[s]["easy"])
        qs = []
        for q in groups:
            i, k = np.nonzero(legal[q])
            if len(i):
                qs.append((q[i], k))
        if not qs:
            raise ValueError("First-query count match requires a legal action")
        result[rec] = dict(site=s, ix=ix, qs=qs, labels=labels, offset=offset, benefit=benefit)
    return ols, result


def training(m, cases, fit):
    donors, count = [], Counter()
    for s in fit:
        rs = []
        for r in cases.values():
            if r["site"] != s:
                continue
            data = []
            for i, k in r["qs"]:
                target = r["labels"][i, k].astype(np.float32)
                data.append((m.x[r["ix"][i], k].astype(np.float32), target))
                count.update(actions=len(i), cv_positive=int(target[:, 0].sum()), dual_positive=int(target[:, 1].sum()))
            rs.append(data)
        donors.append(rs)
    return donors, count


def batcher(donors):
    def batch(rng):
        sizes = rng.multinomial(128, np.full(len(donors), 1./len(donors)))
        x, y = [], []
        for s, n in zip(donors, sizes, strict=True):
            for r in rng.integers(len(s), size=n):
                qs = s[r]
                a, b = qs[rng.integers(len(qs))]
                j = rng.integers(len(a))
                x.append(a[j])
                y.append(b[j])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))
    return batch


def best(i, k, z, allowed):
    keep = np.flatnonzero(allowed)
    if not len(keep):
        return None
    j = keep[z[keep].argmax()]
    return int(i[j]), int(k[j])


@torch.inference_mode()
def score(m, rec, r, ols, y, valid, net):
    ix, s, labels = r["ix"], r["site"], r["labels"]
    c, b = m.floor[ix], m.bank[ix]
    i, k = r["qs"][0]
    x = m.x[ix[i], k].astype(np.float32)
    z = net(torch.from_numpy(x)).numpy()
    u = run.model.decision_costs(m.x[ix], m.co[s], b, c, m.scale[ix])[0]
    every = tuple(np.concatenate(a) for a in zip(*r["qs"], strict=True))
    paths = {run.parent.CV: c, "cv": c, "ols": ols[ix],
             "direct": run.model.selected_paths(m.direct[ix], b, c)}
    picks = {"gain": best(i, k, u[i, k], np.ones(len(i), bool))}
    for j, arm in enumerate(LEARNED):
        picks[arm] = best(i, k, z[:, j], np.ones(len(i), bool))
        for t, (a, f) in (("first", (i, k)), ("window", every)):
            name = "cv" if j == 0 else "dual"
            picks[f"{t}_{name}_oracle"] = best(a, f, r["benefit"][a, f]/m.d["crowd"][ix[a]],
                                               labels[a, f, j])
    for arm, pick in picks.items():
        paths[arm] = c.copy()
        if pick is not None:
            a, f = pick
            paths[arm][a] = b[a, f]
    pairs = {}
    for a in ARMS:
        pairs[a+"_cv"] = ("cv", a, "cv")
        pairs[a+"_ols"] = ("ols", a, "cv")
    pairs.update(cv_head_vs_gain=("gain", "cv_head", "cv"), dual_head_vs_gain=("gain", "dual_head", "cv"))
    cut = run.cuts_by_source()[s]
    out = run.parent.readout.score_recording(paths, pairs, y[ix], valid[ix], m.d["queries"][ix], rec,
                                            easy_cut=cut["easy"], hard_cut=cut["hard"])
    metrics = {}
    for a in ARMS:
        risk = out["metrics"][a]["risk"]
        low_cv, low_ols = (run.parent.readout.merge_comparisons([out["comparisons"][a+"_"+q]])["completion_lower_pp"]
                           for q in ("cv", "ols"))
        cv_joint = risk["supported"] and low_cv is not None and low_cv > 0.
        dual = cv_joint and low_ols is not None and low_ols > 0.
        metrics[a] = dict(edits=risk["selected"], risk_pass=int(risk["supported"]),
                          cv_joint=int(cv_joint), dual_positive=int(dual),
                          undefined=int(risk["completion_easy_risk_upper"] is None), views=1)
        if a in picks:
            pick = picks[a]
            expect = labels[pick[0], pick[1]] if pick is not None else (False, False)
            if tuple(expect) != (cv_joint, dual):
                raise ValueError("One-edit oracle labels disagree with original scoring")
    for a in ("gain", *LEARNED):
        if metrics[a]["edits"] != 1:
            raise ValueError("First-query action count mismatch")
    return dict(metrics=metrics, comparisons=out["comparisons"],
                counts=dict(cv_candidates=int(labels[..., 0].sum()), dual_candidates=int(labels[..., 1].sum()),
                            cv_first=int(labels[i, k, 0].sum()), dual_first=int(labels[i, k, 1].sum()),
                            ols_already_worse_than_cv=int(r["offset"] >= 0.)))


def execute(key, replay, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(("target_eval", "valid", "frames"))
    records = h.previous.prior.blocks(d, lab["frames"])
    y, valid = lab["target_eval"]-d["origin"][:, None], lab["valid"]
    parents, controls, ranks = h.cost.banks()
    trainer.PRIVATE = PRIVATE
    sites = sorted(set(d["sites"]))
    result, freeze, updates = {s: [] for s in sites}, {}, 0
    for cell in parents:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        ols, cases = prepare(m, records, y, valid)
        for ctrl in controls[m.seed]:
            role = ctrl["identity"]
            s, fit, cal = role["held"], role["fit"], role["cal_a"]+role["cal_b"]
            if (len(fit), len(cal), len(set(fit+cal+[s]))) != (7, 6, 14) or set(fit+cal+[s]) != set(sites):
                raise ValueError("Preserve frozen source roles")
            donors, count = training(m, cases, fit)
            name = f"{m.seed}_{s}"
            ident = dict(registration=key, held=s, fit=fit, reserved=cal, seed=m.seed)
            width = m.x.shape[-1]
            net, freeze[name], n = trainer.fit(ident, name, m.seed, lambda: model.network(width), STEPS,
                                              batcher(donors), model.loss, replay, beat)
            updates += n
            scored = [score(m, rec, r, ols, y, valid, net) for rec, r in cases.items() if r["site"] == s]
            result[s].append(dict(seed=m.seed, donor=count,
                counts=sum((Counter(r["counts"]) for r in scored), Counter()),
                metrics={k: {f: sum(r["metrics"][k][f] for r in scored) for f in scored[0]["metrics"][k]} for k in ARMS},
                effects={k: run.parent.readout.merge_comparisons([r["comparisons"][k] for r in scored])["completion_lower_pp"]
                         for k in scored[0]["comparisons"]}))
            beat(phase="held", source=s, seed=m.seed, updates=updates, donor_dual=count["dual_positive"])
        del m
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    metrics = {k: {f: sum(r["metrics"][k][f] for r in flat) for f in flat[0]["metrics"][k]} for k in ARMS}
    if metrics["direct"]["views"] != 102 or len(flat) != 42:
        raise ValueError("Require the complete pedestrian development cohort")
    ref = run.read(parent.PUBLIC/"analysis.json")["risk"]["direct"]
    for a, b in (("risk_pass", "passes"), ("edits", "switches"), ("undefined", "undefined_easy")):
        if metrics["direct"][a] != ref[b]:
            raise ValueError("Dense direct policy failed to replay")
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=dict(metrics=metrics, effects={k: ci(k) for k in flat[0]["effects"]},
                     counts=sum((Counter(r["counts"]) for r in flat), Counter())),
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
