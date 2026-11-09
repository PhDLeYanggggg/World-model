"""Separate utility from risk on unchanged DroneCrowd independent forecasts."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_history_risk_relative_envelope as risk_guard
from src.world_model import m3w_recording_utility_guard as study


stage, run, np = risk_guard.stage, risk_guard.run, risk_guard.np
PUBLIC = stage.PUBLIC.parent/"dronecrowd_recording_utility_guard_v1"
PRIVATE = stage.PRIVATE.parent/PUBLIC.name


def register():
    k, manifest, old, roster, seal, _ = stage.register()
    p = stage.PUBLIC/"analysis.json"
    old_result = run.read(p)
    if old_result["registration"] != k or old_result["confirmation_access"]:
        raise ValueError("Require the completed history-risk study")
    reg = PUBLIC/"registration.json"
    run.once(reg, run.stable(dict(code=[run.artifact(Path(x)) for x in (__file__, study.__file__)],
        parents=[run.artifact(p), run.artifact(stage.PUBLIC/"independent_gate.json")],
        utility="OLS4 error minus frozen independent error on the last four observed positions; "
                "fit paths on the first four; aggregate query means and normalize by CV error",
        target="same twelve-step conservative OLS4 utility in CV-normalized percentage points",
        fit="outer held-source ridge; inner held-source maximum overprediction for lower envelope",
        methods=study.SIGNALS, penalty=study.PENALTY, primary="combined vs donor ridge utility MAE, paired by source",
        guard="history-only risk point and log envelope unchanged; separately require positive OLS4 utility",
        forecasts_changed=False, original_2pct=True, confirmation_access=False)))
    return run.sha(reg), k, manifest, old, roster, seal


def freeze(key, manifest, old, roster, seal):
    meta, provenance, _ = stage.target.context(roster)
    rows = []
    for i, (cell, f) in enumerate(zip(old["clips"], seal["recordings"], strict=True), 1):
        rec = cell["sequence"]
        if f["recording"] != rec:
            raise ValueError("Recording order changed")
        mark = PRIVATE/f"{rec}.json"
        if not mark.exists():
            d = stage.target.old_arrays(cell)
            src = stage.cal.reader(manifest, rec, meta, provenance)
            x = np.load(run.checked(f["choices"]), mmap_mode="r", allow_pickle=False)
            np.testing.assert_array_equal(src.query_frames, d["frames"])
            total, ref = np.zeros(x.shape[1]), 0.
            for frame, lo, hi in zip(d["frames"], d["offsets"][:-1], d["offsets"][1:], strict=True):
                q = src.get_inputs(int(frame))
                at = q["history_complete"]
                h = q["positions"][at]
                np.testing.assert_array_equal(q["agent_ids"][at], d["agents"][lo:hi])
                np.testing.assert_array_equal(h[:, -1], d["origin"][lo:hi])
                a, b = study.prefix_utility(h, x[lo:hi, :, 0])
                total += a
                ref += b
            if ref <= 0.:
                raise ValueError("Need observed CV reference mass")
            run.once(mark, run.stable(dict(registration=key, recording=rec, choices=f["choices"],
                                          gain_pp=(100.*total/ref).tolist())))
        r = run.read(mark)
        if r["registration"] != key or r["recording"] != rec or r["choices"] != f["choices"]:
            raise ValueError("Mismatched prefix utility receipt")
        rows.append(r)
        if i % 10 == 0:
            print(f"{i}/{len(roster)} observed-only utility recordings", flush=True)
    run.once(PUBLIC/"freeze.json", run.stable(dict(registration=key, recordings=rows, future_labels_read=False,
                                                  confirmation_access=False)))


def score(key, prior_key, roster, seal):
    p = PUBLIC/"freeze.json"
    frozen = run.read(p)
    if frozen["registration"] != key or frozen["future_labels_read"] or frozen["confirmation_access"]:
        raise ValueError("Need outcome-free utility features")
    (_, h, risks, d_gain, groups, risk_pass, dual), _ = stage.read_frozen(prior_key, roster, seal)
    point_risk, upper_risk = risk_guard.relative_bounds(h[..., None], risks, groups)
    hist, actual = [], []
    for f, a in zip(seal["recordings"], frozen["recordings"], strict=True):
        r = run.read(stage.parent.PRIVATE/f"{f['recording']}.score.json")
        if r["freeze"] != f or a["recording"] != f["recording"] or a["choices"] != f["choices"]:
            raise ValueError("Scores and utility features must share exact paths")
        hist.extend(a["gain_pp"])
        actual.extend([row[0]["actual"]["ols4"] for row in r["observed"]])
    hist, y = np.asarray(hist), np.asarray(actual)
    gain = d_gain[:, 0]*100.
    names = np.unique(groups)
    x = np.column_stack((gain, hist))
    predictions = {"raw_donor": (gain, None)}
    for k, z in zip(study.SIGNALS, (x[:, :0], x[:, :1], x[:, 1:], x), strict=True):
        predictions[k] = study.folds(z, y, groups)
    draws = np.random.default_rng(stage.target.SEED).integers(len(names), size=(stage.target.DRAWS, len(names)))
    errors = {k: np.array([np.abs(y-v)[groups == g].mean() for g in names]) for k, (v, _) in predictions.items()}
    menu = {}
    for k, (v, low) in predictions.items():
        effect = errors["donor"]-errors[k]
        gates = {}
        for r_name, r in (("risk_point", point_risk), ("risk_log_envelope", upper_risk)):
            gates[r_name] = {name: stage.selections(r[:, :1], a[:, None], risk_pass[:, :1], dual[:, :1], groups)
                             for name, a in (("utility_point", v), ("utility_lower", low)) if a is not None}
        menu[k] = dict(mae_pp=float(errors[k].mean()), mae_reduction_vs_donor_pp=dict(point=float(effect.mean()),
            ci95=np.quantile(effect[draws].mean(1), [.025, .975]).tolist()),
            positive_sign_accuracy=float(np.mean([np.mean(((v > 0.) == (y > 0.))[groups == g]) for g in names])), gates=gates)
    prior = run.read(stage.PUBLIC/"independent_gate.json")["result"]
    if menu["raw_donor"]["gates"]["risk_log_envelope"]["utility_point"] != prior:
        raise ValueError("Existing independent release changed")
    out = dict(registration=key, freeze=run.artifact(p), metrics=menu, recordings=len(frozen["recordings"]),
        views=len(y), sources=len(names), result_source="fresh_run", label_reads="cached recording scores only",
        original_2pct=True, independent_confirmation=False, deployment_changed=False, confirmation_access=False)
    run.once(PUBLIC/"analysis.json", run.stable(out))
    print(menu, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    phase = p.parse_args().phase
    for d in (PUBLIC, PRIVATE):
        d.mkdir(parents=True, exist_ok=True)
    key, prior_key, manifest, old, roster, seal = register()
    if phase == "freeze":
        freeze(key, manifest, old, roster, seal)
    else:
        score(key, prior_key, roster, seal)


if __name__ == "__main__":
    main()
