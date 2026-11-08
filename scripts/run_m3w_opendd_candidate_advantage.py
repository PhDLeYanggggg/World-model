"""Learn candidate advantage using double-excluded OpenDD forecast producers."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import expit

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_hardness_gate as prior
from scripts import run_m3w_opendd_comoving_eqmotion as forecast
from src.world_model import m3w_opendd_candidate_advantage as model

run, parent, paired, two = prior.run, prior.parent, prior.paired, prior.two
OUT = prior.OUT.parent / "opendd_candidate_advantage_v1"
PRIVATE = forecast.PRIVATE.parent / OUT.name
STEPS = 500
ARMS = ("cv", "hardness", "motion", "gain", "risk", "oracle")


def register():
    key, a, _, _, old, source = prior.register()
    p, h = [prior.OUT / f"{k}.json" for k in ("analysis", "head_freeze")]
    report, heads = run.read(p), run.read(h)
    if (any(v["registration"] != key for v in (report, heads)) or report["heads"] != run.sha(h)
            or report["confirmation_access"] or not report["original_2pct"]):
        raise ValueError("Require the verified hardness rule for frozen query counts")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parents=[run.artifact(x) for x in (p, h, forecast.OUT/"analysis.json")],
        architecture=forecast.model.ARCH, producer_steps=forecast.STEPS, producer_batch=forecast.BATCH,
        readout_steps=STEPS, readout_batch=128, native_steps=[8, 12], seeds=two.study.prior.SEEDS,
        crossfit="each fitting-site prediction is produced by a forecaster excluding both that site and the outer evaluation site",
        target="candidate relative to CV: missing-step lower gain, all/easy/signed-easy excess and easy support",
        primary="direct conservative gain vs original hardness rule using the same path and count per query",
        controls="CV, motion disagreement, expected gain only, expected gain with risk penalty, equal-count label-aware utility",
        no_threshold_search=True, original_2pct=True, confirmation_access=False, threads=4, workers=0,
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"))))
    return run.sha(reg), a, report, heads["heads"], old, source


def frozen_prediction(key, excluded, target, seed, pred):
    name = "_".join(excluded)+f"_{seed}_predict_{target}"
    path = PRIVATE / f"{name}.npy"
    if path.exists():
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), pred)
    else:
        tmp = path.with_suffix(".pending.npy")
        np.save(tmp, pred, allow_pickle=False)
        tmp.replace(path)
    f = path.with_suffix(".json")
    run.once(f, run.stable(dict(registration=key, excluded=list(excluded), target=target, seed=seed,
                                prediction=run.artifact(path))))
    return run.read(f)


def producer_bank(key, sites, seeds, replay, beat):
    heads, predictions, updates = {}, {}, 0
    for excluded, donors in model.ordered_pairs(sites):
        batch = forecast.batcher([sites[s] for s in donors])
        for seed in seeds:
            name = "producer_"+"_".join(excluded)+f"_{seed}"
            ident = dict(registration=key, excluded=list(excluded), fit=list(donors), seed=seed)
            net, heads[name], u = two.fit.fit(ident, name, seed, lambda: forecast.model.Forecast("comoving_ego"),
                forecast.STEPS, batch, forecast.model.loss, replay, beat)
            updates += u
            for target in excluded:
                pred = forecast.predict(net, sites[target][3])
                predictions[excluded, target, seed] = frozen_prediction(key, excluded, target, seed, pred)
            beat(phase="producer", excluded=excluded, seed=seed, updates=updates)
    return heads, predictions, updates


def targets(d, cv, path, cut, scale):
    y, valid = d["y"], d["valid"]
    m = two.moments(cv, path, y, valid, cut)
    _, lower = paired(cv, path, y, valid)
    z = np.column_stack((lower.mean(1), m["all_excess"], m["easy_excess"], m["easy_signed"]))
    return np.c_[(z/scale[:, None]).clip(-8., 8.), m["easy"]].astype(np.float32)


def supervised(held, site, seed, sites, banks, predictions, cut):
    if held == site:
        raise ValueError("Held rows cannot be used for selector fitting")
    pair = tuple(sorted((held, site)))
    receipt = predictions[pair, site, seed]
    if receipt["excluded"] != list(pair) or receipt["target"] != site or receipt["seed"] != seed:
        raise ValueError("Incompatible double-excluded teacher receipt")
    d = sites[site][0]
    cv = banks[site][0]
    path = np.load(run.checked(receipt["prediction"]), allow_pickle=False)
    x, scale = model.features(d["h"], cv, path)
    legal = prior.prior.model.peak(path-cv) > 0.
    return torch.from_numpy(x[legal]), torch.from_numpy(targets(d, cv, path, cut, scale)[legal])


def relative_effect(d, left, right):
    # Relative positions need shared observed future support for a pair.
    h, y, valid, q = (d[k] for k in ("h", "y", "valid", "query"))
    edges = []
    for k in np.unique(q):
        at = np.flatnonzero(q == k)
        i, j = np.triu_indices(len(at), 1)
        edges.extend(zip(at[i], at[j], strict=True))
    if not edges:
        return None
    i, j = np.asarray(edges).T
    r = h[i, -1:]-h[j, -1:]
    same = valid[i] & valid[j]
    if not same.any():
        return None
    real = r+y[i]-y[j]
    a, b = (r+p[i]-p[j] for p in (left, right))
    scale = np.linalg.norm(a-real, axis=-1)[same].sum()
    return float(100.*paired(a, b, real, same)[1].sum()/scale) if scale > 0. else None


@torch.inference_mode()
def evaluate(d, bank, ego, cuts, net, hard_head, ref):
    cv, ols = bank[:2]
    features, scale = model.features(d["h"], cv, ego)
    z = net(torch.from_numpy(features)).numpy()
    gain, risk = model.scores(z, scale)
    hard = hard_head(torch.from_numpy(prior.model.features(d["h"]))).softmax(1).numpy()
    legal = prior.prior.model.peak(ego-cv) > 0.
    on = prior.model.decide(hard, legal)
    query = d["query"]
    select = lambda score: prior.model.matched_count(query, on, legal, score)
    picked = dict(hardness=on, motion=select(prior.prior.model.instability(d["h"], cv, ols)),
                  gain=select(gain), risk=select(risk))
    # Forecast scores and all causal choices are fixed before inspecting current outcomes.
    y, valid = d["y"], d["valid"]
    _, true = paired(cv, ego, y, valid)
    row_gain = true.sum(1)
    picked["oracle"] = select(row_gain)
    predicted = {"cv": cv, **{k: np.where(v[:, None, None], ego, cv) for k, v in picked.items()}}
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    pp = lambda a, b: prior.paired_pp(predicted[a], predicted[b], y, valid, mass)
    label, known = prior.labels(d, cv, cuts)
    metric = {}
    for k, p in predicted.items():
        metric[k] = dict(parent.score(d, cv, ols, p, cuts), **prior.count_labels((p != cv).any((1, 2)), label, known),
            total_radius=float(prior.prior.model.peak(p-cv).sum()), cv_lower=pp("cv", k),
            easy_net_percent=None)
        m = two.moments(cv, p, y, valid, cuts[0])
        if m["ref_easy"].sum() > 0.:
            metric[k]["easy_net_percent"] = float(100.*m["easy_signed"].sum()/m["ref_easy"].sum())
    for a, b in (("hardness", "gated"), ("motion", "matched_motion")):
        for k, v in ref["raw"][b].items():
            if k != "lower_vs_cv" and v is not None:
                np.testing.assert_allclose(metric[a][k], v, atol=1e-10, rtol=0.)
        np.testing.assert_allclose(metric[a]["cv_lower"], ref["raw"][b]["lower_vs_cv"], atol=1e-12, rtol=0.)
    pair = {k: relative_effect(d, predicted["hardness"], predicted[k]) for k in ("gain", "risk")}
    classifiers = {k: prior.binary(row_gain[legal] > 0., p[legal]) for k, p in (
        ("hardness", hard[:, 2]), ("gain", gain), ("risk", risk))}
    classifiers["easy_brier"] = float(np.square(expit(z[known, 4])-(label[known] == 0)).mean())
    return dict(metrics=metric, selectors={k: int(v.sum()) for k, v in picked.items()}, diagnostics=classifiers,
        effects=dict(gain_vs_hardness=pp("hardness", "gain"), risk_vs_hardness=pp("hardness", "risk"),
            risk_vs_gain=pp("gain", "risk"), gain_vs_motion=pp("motion", "gain"),
            risk_vs_cv=pp("cv", "risk"), gain_vs_cv=pp("cv", "gain"), oracle_vs_cv=pp("cv", "oracle")),
        relative_vs_hardness=pair)


def execute(key, a, old_hard, old_heads, old, source, replay, beat):
    sites = {s: forecast.load(s, source) for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(v[0]) for s, v in sites.items()}
    seeds = tuple(two.study.prior.SEEDS)
    heads, predictions, updates = producer_bank(key, sites, seeds, replay, beat)
    f = OUT / "producer_freeze.json"
    run.once(f, run.stable(dict(registration=key, heads=heads,
        predictions=[v for _, v in sorted(predictions.items())], outer_sites_excluded=True)))
    result, readers = {}, {}
    cuts = lambda fit: np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in fit]), [.25, .75])
    for held, data in sites.items():
        donors = tuple(s for s in sites if s != held)
        cut = cuts(donors)
        result[held] = []
        for row in a["by_site"][held]:
            seed = row["seed"]
            train = [supervised(held, s, seed, sites, banks, predictions, cuts([u for u in donors if u != s])[0])
                     for s in donors]
            width = train[0][0].shape[1]
            name = f"readout_{held}_{seed}"
            ident = dict(registration=key, held=held, fit=list(donors), seed=seed, producer_manifest=run.sha(f))
            net, readers[name], u = two.fit.fit(ident, name, seed, lambda: model.network(width),
                STEPS, two.batcher(train), model.loss, replay, beat)
            updates += u
            cell = next(v for v in old["by_site"][held] if v["seed"] == seed)
            with np.load(run.checked(cell["prediction"]), allow_pickle=False) as z:
                ego = z["comoving_ego"]
            ref = next(v for v in old_hard["by_site"][held] if v["seed"] == seed)
            previous = old_heads[f"{held}_{seed}"]
            if previous["identity"]["held"] != held or previous["identity"]["fit"] != list(donors):
                raise ValueError("Hardness baseline used different source roles")
            old_width = prior.model.features(data[0]["h"][:1]).shape[1]
            hard = two.fit.load(previous, lambda: prior.model.network(old_width), prior.STEPS)
            result[held].append(dict(seed=seed, **evaluate(data[0], banks[held], ego, cut, net, hard, ref)))
        beat(phase="outer", held=held, updates=updates)
    flat = [r for v in result.values() for r in v]

    def ci(fn):
        cells = [[fn(r) for r in v] for v in result.values()]
        x = np.array([np.mean(v) for v in cells if all(y is not None for y in v)])
        if not len(x):
            return None
        draw = np.random.default_rng(817).integers(len(x), size=(3000, len(x)))
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist(),
                    sites=len(x), worst_site=float(x.min()))

    fields = prior.prior.FIELDS + ("hard", "easy", "unknown", "total_radius")
    summary = dict(policies={k: {f: sum(r["metrics"][k][f] for r in flat) for f in fields} for k in ARMS},
        effects={k: ci(lambda r: r["effects"][k]) for k in flat[0]["effects"]},
        relative={k: ci(lambda r: r["relative_vs_hardness"][k]) for k in ("gain", "risk")},
        decision_diagnostics={k: {f: ci(lambda r: None if r["diagnostics"][k] is None else r["diagnostics"][k][f])
            for f in ("auroc", "auprc", "rate")} for k in ("hardness", "gain", "risk")})
    g = OUT / "head_freeze.json"
    run.once(g, run.stable(dict(registration=key, producers=run.sha(f), heads=readers)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, producers=run.sha(f), heads=run.sha(g),
        summary=summary, by_site=result, original_2pct=True, confirmation_access=False, submission_ready=False)))
    return updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    two.fit.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            event = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", event)
            if v.get("phase") != "fit":
                print(event, flush=True)

        updates = execute(*register(), a.replay, beat)
        beat(phase="complete", updates=updates, replay=a.replay)


if __name__ == "__main__":
    main()
