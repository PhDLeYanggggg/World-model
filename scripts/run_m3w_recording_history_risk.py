"""Test a causal history risk proxy without changing any forecast or intervention."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recording_risk_pool as parent
from src.world_model import m3w_recording_history_risk as study


run, np, cal, target, pool, motion = parent.run, parent.np, parent.cal, parent.target, parent.study, parent.model.floor
PUBLIC = parent.PUBLIC.parent/"dronecrowd_recording_history_risk_v1"
PRIVATE = parent.PRIVATE.parent/PUBLIC.name


def register():
    key, manifest, old, roster, _, _ = parent.register()
    p = parent.PUBLIC/"analysis.json"
    report = run.read(p)
    seal = run.read(run.checked(report["freeze"]))
    if seal["registration"] != key or not seal["outcome_free"] or seal["confirmation_access"] or report["confirmation_access"]:
        raise ValueError("Require the frozen calibration-only menu")
    reg = PUBLIC/"registration.json"
    run.once(reg, run.stable(dict(code=[run.artifact(Path(x)) for x in (__file__, study.__file__)],
        parent=run.artifact(p), signals=study.SIGNALS, penalty=study.PENALTY,
        proxy="fit each analytic path on first four of eight observed positions; score last four; "
              "use the frozen full-history action identities and CV change masks; proxy easy cutoff scaled by 5/13",
        calibration="leave-one-source-out per-arm ridge with equal source weights; "
                    "upper margin uses nested out-of-source residual maxima on the fitting groups",
        primary="held-source risk MAE, combined vs donor-only same ridge; paired source bootstrap",
        choices_changed=False, gain_changed=False, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), manifest, old, roster, seal, report


def compute(src, d, coeff, x, easy_cut):
    np.testing.assert_array_equal(src.query_frames, d["frames"])
    if x.shape != (len(d["cv"]), 3, len(pool.ARMS)) or x.max(initial=0) > 3:
        raise ValueError("Wrong frozen choice array")
    total = np.zeros((*x.shape[1:], 4))
    for f, lo, hi in zip(d["frames"], d["offsets"][:-1], d["offsets"][1:], strict=True):
        q = src.get_inputs(int(f))
        at = q["history_complete"]
        h = q["positions"][at]
        np.testing.assert_array_equal(q["agent_ids"][at], d["agents"][lo:hi])
        np.testing.assert_array_equal(h[:, -1], d["origin"][lo:hi])
        bank = np.concatenate((d["ols"][lo:hi, None], motion.paths(coeff[lo:hi, :3])), axis=1)
        changed = (bank != d["cv"][lo:hi, None]).any((2, 3))
        total += study.prefix_masses(h, x[lo:hi], changed, easy_cut)
    return total.tolist()


def freeze(key, manifest, old, roster, seal):
    meta, provenance, cut = target.context(roster)
    rows = []
    for i, (cell, f) in enumerate(zip(old["clips"], seal["recordings"], strict=True), 1):
        rec = cell["sequence"]
        if rec != f["recording"]:
            raise ValueError("Incorrect recording order")
        mark = PRIVATE/f"{rec}.json"
        if not mark.exists():
            d = target.old_arrays(cell)
            src = cal.reader(manifest, rec, meta, provenance)
            c = np.load(run.checked(f["parent"]["coefficients"]), mmap_mode="r", allow_pickle=False)
            x = np.load(run.checked(f["choices"]), mmap_mode="r", allow_pickle=False)
            val = compute(src, d, c, x, cut["easy"])
            run.once(mark, run.stable(dict(registration=key, recording=rec, frozen_choices=f["choices"], prefix_masses=val)))
        r = run.read(mark)
        if r["registration"] != key or r["recording"] != rec or r["frozen_choices"] != f["choices"]:
            raise ValueError("Wrong proxy receipt")
        rows.append(r)
        if i % 10 == 0:
            print(f"{i}/{len(roster)} observed-prefix recordings", flush=True)
    run.once(PUBLIC/"freeze.json", run.stable(dict(registration=key, recordings=rows,
                                                  future_labels_used=False, confirmation_access=False)))


def selections(pred, gain, risk_pass, dual_pass, groups):
    chosen = [pool.select(r, g, np.zeros_like(r)) for r, g in zip(pred, gain, strict=True)]
    release = [(i, j) for i, j in enumerate(chosen) if j is not None]
    opportunity = dual_pass.any(1)
    gated = ((pred <= pool.TOLERANCE).all(2) & dual_pass).any(1)
    positive = ((pred <= pool.TOLERANCE).all(2) & (gain > 0.) & dual_pass).any(1)
    return dict(selected=len(release), selected_sources=len({groups[i] for i, _ in release}),
        passed_risk=sum(bool(risk_pass[i, j]) for i, j in release),
        passed_both=sum(bool(dual_pass[i, j]) for i, j in release),
        opportunity_gated_by_risk=int((opportunity & ~gated).sum()),
        opportunity_gated_by_gain=int((gated & ~positive).sum()))


def read_frozen(key, roster, parent_seal):
    p = PUBLIC/"freeze.json"
    seal = run.read(p)
    if seal["registration"] != key or seal["future_labels_used"] or seal["confirmation_access"]:
        raise ValueError("Risk proxies must be frozen before scoring")
    donor, hist, actual, gain, groups, passed, dual = [], [], [], [], [], [], []
    sources = []
    for f, h in zip(parent_seal["recordings"], seal["recordings"], strict=True):
        rec = f["recording"]
        if rec != h["recording"] or f["choices"] != h["frozen_choices"]:
            raise ValueError("Proxy and score choices differ")
        r = parent.PRIVATE/f"{rec}.score.json"
        score = run.read(r)
        if score["freeze"] != f:
            raise ValueError("Incorrect precomputed score receipt")
        sources.append(run.artifact(r))
        for d, q, y in zip(f["predicted_recording_masses"], h["prefix_masses"], score["observed"], strict=True):
            d_risk, g = pool.prediction(d)
            q = np.asarray(q)
            if not (q[:, 2:] > 0.).all() or not np.isfinite(d_risk).all():
                raise ValueError("Need support in each frozen recording proxy")
            hist.append(q[:, :2]/q[:, 2:])
            donor.append(d_risk)
            actual.append([[1. if t is None else t for t in row["risk"]] for row in y])
            gain.append(g)
            groups.append(roster[rec])
            passed.append([row["risk_pass"] for row in y])
            dual.append([row["dual_pass"] for row in y])
    return [np.asarray(x) for x in (donor, hist, actual, gain, groups, passed, dual)], sources


def score(key, roster, seal, prior):
    (donor, hist, y, gain, groups, risk_pass, dual), sources = read_frozen(key, roster, seal)
    if dual.any(1).sum() != prior["hindsight_available_joint"]:
        raise ValueError("Forecast opportunity changed")
    names = np.unique(groups)
    residuals = {g: (y-donor)[groups == g] for g in names}
    median, upper = np.zeros_like(y), np.zeros_like(y)
    for g in names:
        _, m, u = pool.group_offsets(residuals, g)
        median[groups == g], upper[groups == g] = donor[groups == g]+m, donor[groups == g]+u
    pred = {"donor_offset": (median, upper)}
    for s, x in zip(study.SIGNALS, (donor[..., None], hist[..., None], np.stack((donor, hist), axis=-1)), strict=True):
        pred[s] = study.source_folds(x, y, groups)
    boot = np.random.default_rng(target.SEED).integers(len(names), size=(target.DRAWS, len(names)))
    err = {k: 100.*np.stack([np.abs(p[0]-y)[groups == g].mean((0, 1)) for g in names]) for k, p in pred.items()}
    out = {}
    for k, (point, upper) in pred.items():
        metrics = dict(mae_all_easy_pp=err[k].mean(0).tolist(),
            point=selections(point, gain, risk_pass, dual, groups), envelope=selections(upper, gain, risk_pass, dual, groups))
        z = (err["donor"]-err[k]).mean(1)
        metrics["mae_reduction_vs_donor_pp"] = dict(point=float(z.mean()), ci95=np.quantile(z[boot].mean(1), [.025, .975]).tolist())
        out[k] = metrics
    for k, old in (("point", "source_median"), ("envelope", "source_envelope")):
        a, b = out["donor_offset"][k], prior["held_source_selection"][old]
        if [a["selected"], a["passed_risk"], a["passed_both"]] != [b["released"], b["passed_risk"], b["passed_both_and_risk"]]:
            raise ValueError("Previous risk gate did not reproduce")
    run.once(PUBLIC/"analysis.json", run.stable(dict(registration=key, freeze=run.artifact(PUBLIC/"freeze.json"),
        result_source="fresh_run", metrics=out, score_sources=sources, views=len(y), sources=len(names),
        opportunities=int(dual.any(1).sum()), label_reads="cached recording scores only", original_2pct=True,
        confirmation_access=False, deployment_changed=False)))
    print(out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    key, manifest, old, roster, seal, prior = register()
    if phase == "freeze":
        freeze(key, manifest, old, roster, seal)
    else:
        score(key, roster, seal, prior)


if __name__ == "__main__":
    main()
