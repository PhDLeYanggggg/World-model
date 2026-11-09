"""Measure observed input shift across the existing donor and calibration rosters."""

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_full_scene_harm_transfer as study


np, run, cal, target, model = study.np, study.run, study.cal, study.target, study.model
FIELDS = ("query_size", "mean_step_px", "cv_prefix_error_in_actor_scale", "ols_cv_peak_in_actor_scale")


def descriptors(h):
    a, _, paths, scale = model.features(h)
    peak = np.linalg.norm(paths[:, 1]-paths[:, 0], axis=-1).max(1)
    return np.column_stack((np.full(len(h), len(h)), scale, a[:, -4:].mean(1), peak/scale))


def quantiles(x):
    return dict(rows=len(x), quantiles=np.quantile(x, [.01, .25, .5, .75, .99], axis=0).T.tolist())


def main():
    key, manifest, old, roster, _, _ = study.register()
    seal = run.read(study.PUBLIC/"freeze.json")
    if seal["registration"] != key or seal["held_labels_used"] or seal["confirmation_access"]:
        raise ValueError("Require the frozen observed-input roster")
    d = run.population()
    train = defaultdict(list)
    for q in d["groups"]:
        train[str(d["sites"][q[0]])].append(descriptors(d["history"][q]))
    meta, provenance, _ = target.context(roster)
    groups = defaultdict(list)
    for i, cell in enumerate(old["clips"], 1):
        rec = cell["sequence"]
        src = cal.reader(manifest, rec, meta, provenance)
        a = target.old_arrays(cell)
        np.testing.assert_array_equal(src.query_frames, a["frames"])
        for f, lo, hi in zip(a["frames"], a["offsets"][:-1], a["offsets"][1:], strict=True):
            x = src.get_inputs(int(f))
            at = x["history_complete"]
            np.testing.assert_array_equal(x["agent_ids"][at], a["agents"][lo:hi])
            groups[roster[rec]].append(descriptors(x["positions"][at]))
        if i % 10 == 0:
            print(f"{i}/{len(roster)} input-only recordings", flush=True)
    left = {k: np.concatenate(v) for k, v in train.items()}
    right = {k: np.concatenate(v) for k, v in groups.items()}
    x, z = (np.concatenate(list(v.values())) for v in (left, right))
    if len(x) != 125540 or len(z) != 1757584 or len(right) != 15 or len(left) != 14:
        raise ValueError("Roster mismatch")
    lo, hi = np.quantile(x, [.01, .99], axis=0)
    out = dict(registration=key, code=run.artifact(Path(__file__)), fields=FIELDS,
        quantile_order=[.01, .25, .5, .75, .99], donor=quantiles(x), calibration=quantiles(z),
        calibration_outside_donor_1_99=np.mean((z < lo) | (z > hi), axis=0).tolist(),
        by_donor={k: quantiles(v) for k, v in left.items()},
        by_calibration_group={k: quantiles(v) for k, v in right.items()},
        weighting="each requested actor-query instance once; descriptive not independent sample counts",
        result_source="fresh_run", calibration_labels_read=False, confirmation_access=False,
        no_model_or_threshold_changes=True)
    run.once(study.PUBLIC/"input_shift.json", run.stable(out))
    print({k: out[k] for k in ("donor", "calibration", "calibration_outside_donor_1_99")})


if __name__ == "__main__":
    main()
