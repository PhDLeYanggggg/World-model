"""Rerank the gain slot while preserving the frozen support forecast."""

import argparse
from collections import Counter
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
from scripts import run_m3w_opendd_slate_queue as study
from scripts import run_m3w_opendd_budget_rank as rank
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_opendd_protected_gain as model

record, run, v, two, hard, prior = rank.record, rank.run, rank.v, rank.two, rank.hard, rank.prior
OUT, PRIVATE = rank.OUT.parent/"opendd_protected_gain_v1", rank.PRIVATE.parent/"opendd_protected_gain_v1"
STEPS = 500
HEADS = ("context", "masked")
ARMS = ("original", "gain_only", "protected", "protected_masked", "guard",
        "oracle_gain", "oracle_protected", "oracle_preserve_reference")


def register():
    key, outer, old, source, gains, prev, producers, releases, nested = rank.register()
    p, h, a, r = (rank.OUT/f"{k}.json" for k in ("analysis", "head_freeze", "attribution_analysis", "attribution_registration"))
    ref, freeze, att, receipt = map(run.read, (p, h, a, r))
    if (ref["registration"] != key or ref["heads"] != run.sha(h) or freeze["registration"] != key
            or att["registration"] != run.sha(r) or receipt["parents"][0]["sha256"] != run.sha(p)
            or any(z["confirmation_access"] or not z["original_2pct"] for z in (ref, att, receipt))):
        raise ValueError("Require the frozen recipient-loss attribution")
    f = OUT/"registration.json"
    run.once(f, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(x) for x in (p, h, a, r)],
        steps=STEPS, seeds=two.study.prior.SEEDS, threads=4, workers=0, native_steps=[8, 12],
        matching="preserve both original query role counts and peak displacement, exact support path and recipient",
        protection="gain recipient can change only when support already emitted or co-observed",
        context="observed support trajectory features and elapsed query count; later support is not an input",
        label="direct paired gain and recording risk, plus loss of the originally defined selected easy reference",
        fitting="same nested exclusion lineage as budget rank; context-masked arm retrained on identical actions",
        primary="protected vs frozen original at 943 matched edits; context ablation and exact menus diagnostic",
        no_threshold_search=True, original_2pct=True, confirmation_access=False,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(f), outer, old, source, gains, prev, producers, releases, nested


def groups(d, cv, ols, path, release, first, analytic):
    q, hist = d["query"], d["h"]
    width = study.c.model.features(hist[:1], cv[:1], cv[:1])[0].shape[1]+2
    for at, frozen, packets in rank.packets(d, cv, ols, path, release, first, analytic):
        anchors = [z for _, actions, _, _, _ in packets for z in actions[0]]
        support = next((z for z in anchors if z[0] == 1), None)
        match = next((z for z in packets if any(a[0] == 0 for a in z[1][0])), None)
        if match is None:
            yield at, frozen, None
            continue
        rows, options, feats, paths, unit = match
        indices = [i for i, p in enumerate(options) if model.same_support(options[0], p)]
        ops = [options[i] for i in indices]
        pred = [paths[i] for i in indices]
        seen = bool(support is not None and q[support[1]] <= q[rows[0]])
        context = np.zeros(width, dtype=np.float32)
        if seen:
            _, j, p = support
            context[:width-2] = study.c.model.features(hist[j:j+1], cv[j:j+1], p[None])[0][0]
            context[-2:] = (1., np.log1p(q[rows[0]]-q[j]))
        x = np.c_[feats[indices], np.broadcast_to(context, (len(ops), width))]
        keep = np.array([model.allowed(ops[0], p, seen) for p in ops])
        yield at, frozen, (rows, ops, pred, x, unit, width, keep, seen)


def reference_losses(d, cv, at, frozen, rows, preds, cut):
    local = np.searchsorted(at, rows)
    y, valid = d["y"][at], d["valid"][at]
    cv0 = cv[at]
    m = two.moments(cv0, cv0, y, valid, cut)
    easy = m["ref_easy"] > 0.
    original = bool((easy & (frozen != cv0).any((1, 2))).any())
    labels = []
    for p in preds:
        z = frozen.copy()
        z[local] = p
        after = bool((easy & (z != cv0).any((1, 2))).any())
        labels.append(original and not after)
    return np.asarray(labels, dtype=bool)


def training(d, cv, ols, path, release, first, analytic, cut):
    xs, ys, width, counts = [], [], None, Counter()
    for at, frozen, menu in groups(d, cv, ols, path, release, first, analytic):
        if menu is None:
            continue
        rows, _, pred, x, unit, width, keep, _ = menu
        z = rank.targets(d, cv, ols, at, frozen, rows, pred, unit, cut)
        y = reference_losses(d, cv, at, frozen, rows, pred, cut)
        xs.append(x)
        ys.append(np.c_[z, y.astype(np.float32)])
        counts.update(alternatives=len(x), protected=int(keep.sum()), reference_loss=int(y.sum()))
    if not xs:
        return None, counts
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys)), width), counts


def view(x, width, mode):
    if mode == "masked":
        x = x.clone()
        x[:, -width:] = 0.
    return x


def oracle(z, keep):
    take = keep & (z[:, 5] > .5)
    return int(np.where(take, z[:, 0]-z[:, 1], -np.inf).argmax()) if take.any() else 0


@torch.inference_mode()
def evaluate(d, bank, path, release, first, analytic, nets, cut, ref):
    cv, ols = bank[:2]
    y, valid, rec, q = (d[k] for k in ("y", "valid", "rec", "query"))
    pred = {k: cv.copy() for k in ARMS}
    counts = Counter()
    for at, frozen, menu in groups(d, cv, ols, path, release, first, analytic):
        for p in pred.values():
            p[at] = frozen
        if menu is None:
            continue
        rows, ops, paths, x, unit, width, keep, seen = menu
        f = torch.from_numpy(x)
        a, b = (nets[k](view(f, width, k)).numpy() for k in HEADS)
        unrestricted = np.ones(len(keep), dtype=bool)
        truth = rank.targets(d, cv, ols, at, frozen, rows, paths, unit, cut[0])
        losses = reference_losses(d, cv, at, frozen, rows, paths, cut[0])
        picks = dict(gain_only=model.choose(a, unrestricted), protected=model.choose(a, keep),
            protected_masked=model.choose(b, keep), guard=model.choose(a, keep, guard=True),
            oracle_gain=oracle(truth, unrestricted), oracle_protected=oracle(truth, keep),
            oracle_preserve_reference=oracle(truth, keep & ~losses))
        old_gain = next((j, p) for r, j, p in ops[0] if r == 0)
        old_radius = rank.shape.peak((old_gain[1]-cv[old_gain[0]])[None])[0]
        counts.update(gain_queries=1, support_seen=int(seen), candidates=len(keep),
                      protected_candidates=int(keep.sum()), gain_menu_ref_loss=int(losses.any()),
                      protected_menu_ref_loss=int((losses & keep).any()))
        for k, i in picks.items():
            j, p = next((j, p) for r, j, p in ops[i] if r == 0)
            if not model.same_support(ops[0], ops[i]) or q[j] != q[old_gain[0]]:
                raise ValueError("Changed the frozen support or gain time")
            np.testing.assert_allclose(rank.shape.peak((p-cv[j])[None])[0], old_radius, rtol=1e-10, atol=1e-12)
            pred[k][rows] = paths[i]
            counts[k+"_changes"] += int(i != 0)
            counts[k+"_recipient_changes"] += int(j != old_gain[0])
            if k.startswith("oracle") and truth[i, 5] > .5:
                score = v.parent.score(dict(y=y[at], valid=valid[at], rec=rec[at]),
                                       cv[at], ols[at], pred[k][at], cut)
                if score["dual_positive"] != 1:
                    raise ValueError("Feasible witness failed original scorer")
    stats = {k: v.parent.score(d, cv, ols, p, cut) for k, p in pred.items()}
    if any(stats["original"][f] != ref["metrics"]["two_roles"][f] for f in v.prior.FIELDS):
        raise ValueError("Original causal allocation did not replay")
    qcount = lambda p: np.bincount(q[(p != cv).any((1, 2))], minlength=q.max()+1)
    for p in pred.values():
        np.testing.assert_array_equal(qcount(p), qcount(pred["original"]))
    risks = {k: policy(p, cv, cv, y, valid, rec, easy_cut=cut[0], hard_cut=cut[1])["recordings"]
             for k, p in pred.items()}
    for k in ARMS[1:]:
        for r, z in risks["original"].items():
            before = z["completion_easy_risk_upper"] is not None
            after = risks[k][r]["completion_easy_risk_upper"] is not None
            counts[k+"_lost_reference"] += int(before and not after)
            counts[k+"_new_reference"] += int(not before and after)
    if counts["oracle_preserve_reference_lost_reference"]:
        raise ValueError("Reference-preserving witness moved a defined recording")
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pairs = [(k, "cv") for k in ARMS]+[(k, "original") for k in ARMS[1:]]
    pairs += [("protected", "gain_only"), ("protected", "protected_masked"), ("guard", "protected")]
    effects = {f"{a}_vs_{b}": hard.paired_pp(cv if b == "cv" else pred[b], pred[a], y, valid, mass)
               for a, b in pairs}
    return dict(metrics=stats, effects=effects, diagnostics=counts)


def execute(key, outer, old, source, gains, prev, producers, releases, nested, replay, beat):
    sites = {s: study.c.forecast.load(s, source)[0] for s in v.parent.source.SITES}
    banks = {s: v.parent.prior.analytic(d) for s, d in sites.items()}
    analytic = {s: rank.shape.analytic(d["h"])[2] for s, d in sites.items()}
    refs = {(tuple(r["excluded"]), r["target"], r["seed"]): r for r in producers}
    cuts = lambda use: np.quantile(np.concatenate([sites[s]["cut_samples"] for s in use]), [.25, .75])
    hard_width = hard.model.features(next(iter(sites.values()))["h"][:1]).shape[1]
    result, freeze, updates = {}, {}, 0

    def first_head(meta, d, cv, path):
        n = prior.feature({k: d[k][:1] for k in ("h", "query")}, cv[:1], path[:1])[0].shape[1]
        return two.fit.load(meta, lambda: prior.model.network(n), prior.STEPS)

    for held, d in sites.items():
        use = [s for s in sites if s != held]
        result[held] = []
        for seed in two.study.prior.SEEDS:
            fit, donor = [], Counter()
            for s in use:
                pair = tuple(sorted((held, s)))
                rr = releases["release_"+"_".join(pair)+f"_{seed}"]
                ff = nested[f"first_{held}_{s}_{seed}"]
                if (set(rr["identity"]["fit"]) & set(pair) or set(ff["identity"]["excluded"]) != set(pair)
                        or set(ff["identity"]["fit"]) & set(pair)):
                    raise ValueError("Nomination models expose held or donor labels")
                release = two.fit.load(rr, lambda: hard.model.network(hard_width), hard.STEPS)
                path = record.read_prediction(refs, pair, s, seed)
                cv, ols = banks[s][:2]
                z, counts = training(sites[s], cv, ols, path, release, first_head(ff, sites[s], cv, path),
                                     analytic[s], cuts([t for t in use if t != s])[0])
                donor.update(counts)
                if z is not None:
                    fit.append(z)
            name = f"{held}_{seed}"
            n = fit[0][0].shape[1]
            nets = {}
            for k in HEADS:
                ident = dict(registration=key, held=held, fit=use, seed=seed, arm=k)
                tag = f"{name}_{k}"
                batch = two.batcher([(view(x, width, k), y) for x, y, width in fit])
                nets[k], freeze[tag], u = two.fit.fit(ident, tag, seed, lambda: model.network(n), STEPS,
                                                       batch, model.loss, replay, beat)
                updates += u
            lookup = lambda a: next(z for z in a["by_site"][held] if z["seed"] == seed)
            release = two.fit.load(outer[name], lambda: hard.model.network(hard_width), hard.STEPS)
            with np.load(run.checked(lookup(old)["prediction"]), allow_pickle=False) as z:
                path = z["comoving_ego"]
            first = first_head(gains[name], d, banks[held][0], path)
            z = evaluate(d, banks[held], path, release, first, analytic[held], nets, cuts(use), lookup(prev))
            result[held].append(dict(seed=seed, donor=donor, **z))
            beat(phase="site", site=held, seed=seed, updates=updates, alternatives=donor["alternatives"])
    flat = [r for rows in result.values() for r in rows]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(k):
        x = np.array([np.mean([r["effects"][k] for r in rows]) for rows in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    worst_site=float(x.min()))

    summary = dict(metrics={k: {f: sum(r["metrics"][k][f] for r in flat) for f in v.prior.FIELDS} for k in ARMS},
        effects={k: ci(k) for k in flat[0]["effects"]}, diagnostics=sum((r["diagnostics"] for r in flat), Counter()))
    f = OUT/"head_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=freeze)))
    run.once(OUT/"analysis.json", run.stable(dict(registration=key, heads=run.sha(f), by_site=result,
        summary=summary, original_2pct=True, confirmation_access=False, submission_ready=False)))
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

        two.fit.PRIVATE = PRIVATE
        u = execute(*register(), a.replay, beat)
        beat(complete=True, updates=u, replay=a.replay)


if __name__ == "__main__":
    main()
