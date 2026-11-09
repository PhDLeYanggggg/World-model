"""Bound utility at the frozen noise-augmented scene policy counts."""

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_observation_augmentation_transfer as study
from src.evaluation.m3w_perfect_switch_ceiling import capped_masses
from src.evaluation.m3w_selection_forecast_readout import errors


np, run, cal, target = study.np, study.run, study.cal, study.target


def main():
    key, manifest, old, roster, *_ = study.register()
    p = study.PUBLIC/"analysis.json"
    result = run.read(p)
    freeze = run.read(run.checked(result["freeze"]))
    if (key != result["registration"] or key != freeze["registration"] or freeze["labels_used"]
            or result["confirmation_access"] or freeze["confirmation_access"]):
        raise ValueError("Require frozen, scored calibration forecasts")
    meta, provenance, _ = target.context(roster)
    groups, total = defaultdict(list), np.zeros(len(study.SEEDS), dtype=np.int64)
    for i, (cell, f) in enumerate(zip(old["clips"], freeze["records"], strict=True), 1):
        rec = cell["sequence"]
        if f["recording"] != rec or rec not in roster:
            raise ValueError("Unexpected calibration recording")
        d = target.old_arrays(cell)
        x = np.load(run.checked(f["choices"]), mmap_mode="r", allow_pickle=False)
        if x.shape != (len(d["cv"]), len(study.SEEDS), len(study.ARMS)):
            raise ValueError("Misaligned frozen choices")
        offsets = d["offsets"]
        edits = np.add.reduceat(x[:, :, study.ARMS.index("noise_scene")] != 0, offsets[:-1], dtype=np.int64)
        total += edits.sum(0)
        src = cal.reader(manifest, rec, meta, provenance)
        y, valid = cal.label_arrays(src, d)
        cv, _ = errors(d["cv"], y, valid)
        upper = capped_masses(np.where(valid.any(1), cv, 0.), offsets, edits)
        w = np.repeat(1./np.diff(offsets), np.diff(offsets))
        m = study.plain.comparison_masses(d["cv"], d["ols"], d["cv"], y, valid, w)
        groups[roster[rec]].append((m["reference"], m["known_delta"], upper.mean()))
        if i % 10 == 0:
            print(f"{i}/{len(roster)} fixed-count bounds", flush=True)
    ref = result["metrics"]["noise_scene"]
    if int(total.sum()) != ref["edits"]:
        raise ValueError("Coverage must match the actual score")
    values = {}
    for g, rows in sorted(groups.items()):
        mass, ols, upper = np.sum(rows, axis=0)
        if mass <= 0.:
            raise ValueError("CV mass missing in required group")
        values[g] = 100.*(upper-ols)/mass
    z = np.asarray(list(values.values()))
    if z.mean()+1.e-9 < ref["lower_vs_ols4_pp"]["point"]:
        raise ValueError("Observed improvement exceeds the perfect-switch bound")
    boot = np.random.default_rng(target.SEED).integers(len(z), size=(target.DRAWS, len(z)))
    out = dict(source=run.artifact(p), code=run.artifact(Path(__file__)), result_source="fresh_run",
        edits_by_seed=total.tolist(), upper_vs_ols4_pp=float(z.mean()),
        ci95=np.quantile(z[boot].mean(1), [.025, .975]).tolist(), groups=values,
        negative_groups=int((z < 0.).sum()), hindsight_only=True, confirmation_access=False,
        interpretation="Perfect forecasts on any actors at the same query counts; ignore risk and omit OLS comparison debit")
    run.once(study.PUBLIC/"coverage.json", run.stable(out))
    print(out["upper_vs_ols4_pp"], out["ci95"], out["negative_groups"], flush=True)


if __name__ == "__main__":
    main()
