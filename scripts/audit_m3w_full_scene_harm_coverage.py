"""Upper-bound any forecaster at the learned policies' frozen per-query counts."""

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_full_scene_harm_transfer as study
from src.evaluation.m3w_perfect_switch_ceiling import capped_masses
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_selection_forecast_readout import errors


cal, run, np, source = study.cal, study.run, study.np, study.target
OUT, LOCAL = study.PUBLIC, study.PRIVATE


def read_recording(manifest, cell, receipt, meta, provenance):
    rec = cell["sequence"]
    reader = cal.reader(manifest, rec, meta, provenance)
    d = source.old_arrays(cell)
    y, v = cal.label_arrays(reader, d)
    c, o, at = (d[k] for k in ("cv", "ols", "offsets"))
    x = np.load(run.checked(receipt["choices"]), allow_pickle=False, mmap_mode="r")
    shape = (len(study.SEEDS), len(study.ARMS))
    if x.shape != (len(c), *shape):
        raise ValueError("Choice and target rosters differ")
    caps = np.add.reduceat((x != 0).astype(np.int16), at[:-1])
    np.testing.assert_array_equal(caps[:, :, study.ARMS.index("scene_budget")],
                                  caps[:, :, study.ARMS.index("matched_gain")])
    ade, _ = errors(c, y, v)
    best = np.where(v.any(1), ade, 0.)
    weight = np.repeat(1./np.diff(at), np.diff(at))
    m = comparison_masses(c, o, c, y, v, weight)
    upper = capped_masses(best, at, caps.reshape(len(at)-1, -1)).reshape(shape)
    return dict(recording=rec, cv_reference=m["reference"], ols_observed_gain=m["known_delta"],
                upper=upper.tolist(), edits=caps.sum(0).tolist())


def summarize(rows, roster, result):
    groups = sorted(set(roster.values()))
    draws = np.random.default_rng(source.SEED).integers(len(groups), size=(source.DRAWS, len(groups)))
    values, edits = {}, np.sum([r["edits"] for r in rows], axis=0)
    for g in groups:
        sub = [r for r in rows if roster[r["recording"]] == g]
        mass = sum(r["cv_reference"] for r in sub)
        ref = sum(r["ols_observed_gain"] for r in sub)
        top = np.sum([r["upper"] for r in sub], axis=0).mean(0)
        if mass <= 0.:
            raise ValueError("CV mass missing in required group")
        values[g] = dict(zip(study.ARMS, np.column_stack((100.*top/mass, 100.*(top-ref)/mass)).tolist(), strict=True))
    metrics = {}
    for i, k in enumerate(study.ARMS):
        n = int(edits[:, i].sum())
        if n != result["metrics"][k]["edits"]:
            raise ValueError("Upper bound count differs from realized edits")
        z = np.asarray([r[k] for r in values.values()])
        if z[:, 0].mean()+1e-9 < result["metrics"][k]["lower_gain_vs_cv_pp"]["point"]:
            raise ValueError("Realized gain exceeds perfect-switch bound")
        metrics[k] = dict(edits_by_seed=edits[:, i].tolist(),
            cv_upper_pp=float(z[:, 0].mean()), ols4_upper_pp=float(z[:, 1].mean()),
            ci95=np.quantile(z[draws].mean(1), [.025, .975], axis=0).T.tolist(),
            negative_ols_groups=int((z[:, 1] < 0.).sum()))
    return dict(metrics=metrics, groups=values, seeds=list(study.SEEDS), bound_order=["vs_cv", "vs_ols4"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    with (LOCAL/"cap.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, manifest, old, roster, _, _ = study.register()
        result_path = OUT/"analysis.json"
        result = run.read(result_path)
        seal = run.read(run.checked(result["freeze"]))
        if (key != result["registration"] or key != seal["registration"] or seal["held_labels_used"]
                or result["confirmation_access"] or seal["confirmation_access"]):
            raise ValueError("Only completed approved calibration forecasts can be bounded")
        p = OUT/"coverage_registration.json"
        run.once(p, run.stable(dict(code=run.artifact(Path(__file__)), source=run.artifact(result_path),
            interpretation="perfect predictions at the exact learned intervention counts; ignores risk and omits OLS comparison debit",
            original_rule_changed=False, confirmation_access=False)))
        token = run.sha(p)
        meta, provenance, _ = source.context(roster)
        rows = []
        for i, (cell, f) in enumerate(zip(old["clips"], seal["records"], strict=True), 1):
            rec = cell["sequence"]
            if f["recording"] != rec or rec not in roster:
                raise ValueError("Wrong registered calibration recording")
            mark = LOCAL/f"{rec}.coverage.json"
            if not mark.exists():
                if args.replay:
                    raise ValueError("Replay cannot reopen labels")
                row = read_recording(manifest, cell, f, meta, provenance)
                run.once(mark, run.stable(dict(registration=token, choices=f["choices"], row=row)))
            receipt = run.read(mark)
            if receipt["registration"] != token or receipt["choices"] != f["choices"]:
                raise ValueError("Invalid count bound checkpoint")
            rows.append(receipt["row"])
            study.beat("coverage_replay" if args.replay else "coverage", recording=rec, completed=i, total=len(roster))
        out = dict(registration=token, **summarize(rows, roster, result), result_source="fresh_run",
                   hindsight_only=True, confirmation_access=False, independent_confirmation=False)
        run.once(OUT/"coverage.json", run.stable(out))
        print(out["metrics"], flush=True)


if __name__ == "__main__":
    main()
