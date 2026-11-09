"""Check the input effect of frozen augmentation without opening outcomes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_full_scene_harm_input_shift as audit
from scripts import run_m3w_observation_augmentation_transfer as study


np, run, cal, target = study.np, study.run, study.cal, study.target


def donor_inputs(d, seed):
    # Recreate the registered training stream independently of the label loader.
    rng = np.random.default_rng(seed+27053)
    rows = {k: [] for k in study.MODES}
    for q in d["groups"]:
        h = d["history"][q]
        if rng.random() >= study.aug.KEEP_NATIVE:
            factor = 2.**rng.uniform(study.aug.MIN_POWER, 0.)
            jitter = rng.normal(0., study.aug.NOISE_SD, size=h.shape)
            future = np.zeros((len(h), 12, 2))
            valid = np.zeros((len(h), 12), dtype=bool)
            for k in study.MODES:
                x, _ = study.aug.transform(h, future, valid, factor, jitter if k == "noise" else None)
                rows[k].append(audit.descriptors(x))
        else:
            x = audit.descriptors(h)
            for k in study.MODES:
                rows[k].append(x)
    return {k: np.concatenate(v) for k, v in rows.items()}


def main():
    key, manifest, old, roster, *_ = study.register()
    p = study.PUBLIC/"freeze.json"
    freeze = run.read(p)
    if (freeze["registration"] != key or freeze["labels_used"] or freeze["confirmation_access"]
            or [x["recording"] for x in freeze["records"]] != [x["sequence"] for x in old["clips"]]):
        raise ValueError("Require the frozen roster")
    old_audit = study.plain.PUBLIC/"input_shift.json"
    before = run.read(old_audit)
    if tuple(before["fields"]) != audit.FIELDS or before["calibration_labels_read"] or before["confirmation_access"]:
        raise ValueError("Incorrect original descriptors")
    meta, provenance, _ = target.context(roster)
    values = []
    for i, (cell, frozen) in enumerate(zip(old["clips"], freeze["records"], strict=True), 1):
        rec = cell["sequence"]
        src, prior = cal.reader(manifest, rec, meta, provenance), target.old_arrays(cell)
        np.testing.assert_array_equal(src.query_frames, prior["frames"])
        rows = []
        for f, lo, hi in zip(prior["frames"], prior["offsets"][:-1], prior["offsets"][1:], strict=True):
            x = src.get_inputs(int(f))
            at = x["history_complete"]
            np.testing.assert_array_equal(x["agent_ids"][at], prior["agents"][lo:hi])
            rows.append(audit.descriptors(x["positions"][at]))
        values.append(np.concatenate(rows))
        if len(values[-1]) != frozen["rows"]:
            raise ValueError("Recording row mismatch")
        if i % 10 == 0:
            print(f"{i}/{len(roster)} past-only recordings", flush=True)
    y = np.concatenate(values)
    yy = audit.quantiles(y)
    if yy != before["calibration"]:
        raise ValueError("Target descriptors changed")
    d = run.population()
    arms = {}
    for seed in study.SEEDS:
        for k, x in donor_inputs(d, seed).items():
            if len(x) != before["donor"]["rows"]:
                raise ValueError("Incorrect augmented donor count")
            lo, hi = np.quantile(x, [.01, .99], axis=0)
            arms.setdefault(k, {})[str(seed)] = dict(donor=audit.quantiles(x),
                calibration_outside_donor_1_99=np.mean((y < lo) | (y > hi), axis=0).tolist())
    out = dict(code=run.artifact(Path(__file__)), freeze=run.artifact(p), original_audit=run.artifact(old_audit),
        fields=audit.FIELDS, calibration=yy, plain=before["calibration_outside_donor_1_99"], augmented=arms,
        result_source="fresh_run", calibration_labels_read=False, confirmation_access=False)
    run.once(study.PUBLIC/"input_shift.json", run.stable(out))


if __name__ == "__main__":
    main()
