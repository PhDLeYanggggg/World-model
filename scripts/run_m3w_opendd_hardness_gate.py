"""Test a causal CV-difficulty gate for the frozen comoving candidate."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

if __name__ == "__main__" and (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    raise SystemExit("Use native macOS arm64 .venv-pytorch")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_component_guard as prior
from src.world_model import m3w_opendd_hardness_gate as model

run, parent, paired, ledger, two = prior.run, prior.parent, prior.paired, prior.ledger, prior.two
OUT = prior.OUT.parent / "opendd_hardness_gate_v1"
PRIVATE = prior.prior.PRIVATE.parent / OUT.name
STEPS = 400


def register():
    key, a, ph, h, _, previous, source = prior.register()
    f = prior.OUT / "analysis.json"
    audit = run.read(f)
    if audit["registration"] != key or not audit["original_2pct"] or audit["confirmation_access"]:
        raise ValueError("Require original development scope")
    reg = OUT / "registration.json"
    run.once(reg, run.stable(dict(code=run.artifact(Path(__file__)), model=run.artifact(Path(model.__file__)),
        trainer=run.artifact(Path(two.fit.__file__)), parent=run.artifact(f), native_steps=[8, 12],
        seeds=two.study.prior.SEEDS, steps=STEPS, threads=4, workers=0,
        training="CV easy/other/hard targets only; exclude held site and exclude each fitting site's own labels from cut estimation",
        policy="hard probability >= .50 and easy probability < .25; fixed before evaluation, no threshold search",
        control="same neural trajectory and per-query switch count; replace ranking with observed CV/OLS disagreement",
        primary="direct conservative gain vs matched count control and versus CV",
        fixed_slot_secondary="switch donor-prefix shape to neural at same frozen slot radius if gate passes",
        bootstrap=dict(draws=3000, seed=817, unit="site after averaging seeds"),
        original_2pct=True, confirmation_access=False)))
    return run.sha(reg), a, ph, h, previous, source


def labels(d, cv, cuts):
    m = two.moments(cv, cv, d["y"], d["valid"], cuts[0])
    return np.where(m["easy"], 0, np.where(m["base"] > cuts[1], 2, 1)), m["known"]


def paired_pp(ref, pred, y, valid, mass):
    return float(100.*paired(ref, pred, y, valid)[1].sum()/mass)


def count_labels(on, label, known):
    return dict(hard=int((on & known & (label == 2)).sum()), easy=int((on & known & (label == 0)).sum()),
                unknown=int((on & ~known).sum()), known=int((on & known).sum()))


def binary(y, p):
    if np.unique(y).size < 2:
        return None
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                rate=float(y.mean()))


@torch.inference_mode()
def readout(d, b, x, drift, ego, head, cuts, packet, menu, count, ref):
    cv, ols = b[:2]
    proba = head(x).softmax(1).numpy()
    eligible = prior.model.peak(ego-cv) > 0.
    on = model.decide(proba, eligible)
    eq = model.matched_count(d["query"], on, eligible, drift)
    selected = dict(gated=on, matched_motion=eq)
    raw = {"cv": cv, "full": ego,
           **{k: np.where(v[:, None, None], ego, cv) for k, v in selected.items()}}
    y, valid = d["y"], d["valid"]
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    label, known = labels(d, cv, cuts)
    hard = label[known] == 2
    diag = dict(hard=binary(hard, proba[known, 2]), motion=binary(hard, drift[known]),
                easy=binary(label[known] == 0, proba[known, 0]))
    metrics = {}
    for k, f in raw.items():
        score = parent.score(d, cv, ols, f, cuts)
        applied = (f != cv).any((1, 2))
        m = two.moments(cv, f, y, valid, cuts[0])
        easy_mass = m["ref_easy"].sum()
        metrics[k] = dict(score, **count_labels(applied, label, known),
            total_radius=float(prior.model.peak(f-cv).sum()),
            lower_vs_cv=paired_pp(cv, f, y, valid, mass),
            easy_net_percent=float(100.*m["easy_signed"].sum()/easy_mass) if easy_mass > 0. else None)

    active = np.arange(2) < count[:, None]
    at, arm = menu["row"][active], menu["arm"][active]
    base = packet["paths"][at, arm, 3]
    projection = prior.prior.radius.match_radius(cv[at], base, (ego[at]-cv[at])[:, None])[:, 0]
    anchor, switched = cv.copy(), cv.copy()
    anchor[at] = base
    switched[at] = np.where(on[at, None, None], projection, base)
    np.testing.assert_array_equal(np.flatnonzero((switched != cv).any((1, 2))), np.sort(at))
    np.testing.assert_allclose(prior.model.peak(switched[at]-cv[at]), prior.model.peak(base-cv[at]), atol=1e-10)
    m0, m1 = [parent.score(d, cv, ols, f, cuts) for f in (anchor, switched)]
    for k, v in ref["matched"]["policies"]["prefix_match"].items():
        np.testing.assert_allclose(m0[k], v, atol=1e-10, rtol=0.)
    return dict(raw=metrics, diagnostics=diag, rows=len(cv), shape_switches=int(on[at].sum()),
        slot=dict(prefix=m0, gated=m1),
        effects=dict(raw_gated_vs_motion=paired_pp(raw["matched_motion"], raw["gated"], y, valid, mass),
            raw_gated_vs_cv=metrics["gated"]["lower_vs_cv"], fixed_slot=paired_pp(anchor, switched, y, valid, mass)))


def execute(key, a, ph, h, previous, source, replay, beat):
    sites = {s: prior.prior.load(s, source)[:3] for s in parent.source.SITES}
    banks = {s: parent.prior.analytic(v[0]) for s, v in sites.items()}
    x = {s: torch.from_numpy(model.features(v[0]["h"])) for s, v in sites.items()}
    cuts = lambda selected: np.quantile(np.concatenate([sites[s][0]["cut_samples"] for s in selected]), [.25, .75])
    result, heads, updates = {}, {}, 0
    for held, (d, rec, starts) in sites.items():
        donors = [s for s in sites if s != held]
        cut = cuts(donors)
        train = []
        for s in donors:
            y, known = labels(sites[s][0], banks[s][0], cuts([v for v in donors if v != s]))
            train.append((x[s][known], torch.from_numpy(y[known])))
        fit = two.batcher(train)
        b = banks[held]
        packet = prior.prior.prior.prepare({s: v[0] for s, v in sites.items()}, held, donors, b)
        drift = prior.model.instability(d["h"], *b[:2])
        result[held] = []
        for cell in a["by_site"][held]:
            seed = cell["seed"]
            name = f"{held}_{seed}"
            ident = dict(registration=key, held=held, fit=donors, seed=seed)
            head, heads[name], u = two.fit.fit(ident, name, seed, lambda: model.network(x[held].shape[1]),
                                               STEPS, fit, model.loss, replay, beat)
            updates += u
            ref = next(r for r in previous["by_site"][held] if r["seed"] == seed)
            with np.load(run.checked(ref["prediction"]), allow_pickle=False) as z:
                ego = z["comoving_ego"]
            m, n = ledger.frozen_count(d, b, rec, starts, ph["heads"][f"{held}/{seed}/additive_rank"],
                                      h["heads"][f"{held}_{seed}_motion"])
            result[held].append(dict(seed=seed, **readout(d, b, x[held], drift, ego, head, cut, packet, m, n, ref)))
        beat(phase="held", site=held, updates=updates)
    flat = [r for v in result.values() for r in v]
    draw = np.random.default_rng(817).integers(len(sites), size=(3000, len(sites)))

    def ci(fn):
        x = np.array([np.mean([fn(r) for r in v]) for v in result.values()])
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    totals = {}
    for k in ("raw", "slot"):
        fields = prior.FIELDS + (("hard", "easy", "unknown", "known", "total_radius") if k == "raw" else ())
        totals[k] = {a: {f: sum(r[k][a][f] for r in flat) for f in fields} for a in flat[0][k]}
    totals["effects"] = {k: ci(lambda r: r["effects"][k]) for k in flat[0]["effects"]}
    totals["shape_switches"] = sum(r["shape_switches"] for r in flat)
    totals["diagnostics"] = {}
    for k in flat[0]["diagnostics"]:
        valid = [v for v in result.values() if all(r["diagnostics"][k] is not None for r in v)]
        totals["diagnostics"][k] = dict(sites=len(valid), **{
            f: float(np.mean([np.mean([r["diagnostics"][k][f] for r in v]) for v in valid]))
            for f in ("auroc", "auprc", "rate")})
    freeze = OUT / "head_freeze.json"
    run.once(freeze, run.stable(dict(registration=key, heads=heads)))
    run.once(OUT / "analysis.json", run.stable(dict(registration=key, heads=run.sha(freeze), by_site=result,
        summary=totals, original_2pct=True, confirmation_access=False, submission_ready=False)))
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
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            run.kin.prior.source.atomic(PRIVATE / "heartbeat.json", z)
            if v.get("step", STEPS) == STEPS:
                print(z, flush=True)

        updates = execute(*register(), a.replay, beat)
        beat(complete=True, updates=updates, replay=a.replay)


if __name__ == "__main__":
    main()
