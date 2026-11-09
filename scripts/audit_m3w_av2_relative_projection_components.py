"""Attribute the already-scored relative-minus-individual contrast."""

from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_relative_projection as run

np, study = run.np, run.study


def pair_components(after, before, y, v):
    i, j = np.triu_indices(len(y), 1)
    valid = v[i] & v[j]
    truth = np.where(valid[..., None], y[i]-y[j], 0.)
    a, b = after[i]-after[j], before[i]-before[j]
    norm = lambda z: np.linalg.norm(z, axis=-1)
    known = np.where(valid, norm(b-truth)-norm(a-truth), 0.).sum()/valid.size
    debit = np.where(valid, 0., norm(a-b)).sum()/valid.size
    return float(known), float(debit), float(known-debit)


def main():
    run.torch.set_num_threads(4)
    run.torch.set_num_interop_threads(1)
    key, data, src, _, _, outer, control, _, _, _ = run.register()
    path = run.OUT/"summary.json"
    summary = study.read(path)
    freeze = study.read(study.checked(summary["freeze"]))
    if (freeze["registration"] != key or freeze["held_labels_used"] or summary["independent_confirmation"]
            or summary["reserved_outcomes_opened"]):
        raise ValueError("Require scored held development forecasts")
    rows = []
    for name, ref in freeze["folds"].items():
        f = study.read(study.checked(ref))
        parent = study.read(study.checked(f["identity"]["parent"]))
        r = study.read(run.LOCAL/f"{name}.held.score.json")
        if r["freeze"] != ref or r["role"] != "held":
            raise ValueError("Wrong scoring receipt")
        names = [z["clip"] for z in f["rows"]["held"]]
        city = f["identity"]["held"]
        if city in [f["identity"]["calibration"], *f["identity"]["donors"]] or any(data[n]["city"] != city for n in names):
            raise ValueError("Wrong held label roster")
        truth = run.prev.load_targets(src, data, names)
        for (n, d, soft, ix, digest, _), z, s in zip(
                run.inputs(data, parent, control["folds"][name], outer["folds"][name], "held"),
                f["rows"]["held"], r["rows"], strict=True):
            if not n == z["clip"] == s["clip"] or z["hash"] != digest or z["edited"] != ix.tolist():
                raise ValueError("Wrong frozen paths")
            cv = d["cv"]
            y, v = truth[n]
            paths = np.broadcast_to(cv, (3, *cv.shape)).copy()
            paths[:, ix] = z["paths"]
            trajectory = run.timing.cost.audit.direct_trajectory(paths[2], paths[1], cv, y, v)
            pair = pair_components(paths[2], paths[1], y, v)
            a = s["arms"]["relative"]
            if not np.isclose(pair[2], a["pair_vs_individual"], rtol=0., atol=1e-12) or (
                    trajectory is not None and not np.isclose(trajectory[2], a["vs_individual"], rtol=0., atol=1e-12)):
                raise ValueError("Component identity failed")
            rows.append(dict(city=city, clip=str(n), complete=bool(v.all()), pair=pair, trajectory=trajectory,
                reference_pair=pair_components(paths[0], soft, y, v),
                reference_trajectory=run.timing.cost.audit.direct_trajectory(paths[0], soft, cv, y, v),
                reference_qualification=int(s["arms"]["frozen_position"]["eligible"])-int(s["arms"]["soft"]["eligible"])))
    if len(rows) != summary["held"]["views"]:
        raise ValueError("Incorrect held count")
    stat = {k: {m: run.av.interval(rows, lambda r: r[k][i] if r[k] is not None else None)
                for i, m in enumerate(("known", "missing_debit", "lower"))} for k in ("pair", "trajectory")}
    full = [r for r in rows if r["complete"]]
    out = dict(source=study.artifact(path), result_source="fresh_run", views=len(rows), complete_views=len(full),
        relative_minus_individual=stat, complete_relative_minus_individual={k: run.av.interval(full,
            lambda r: r[k][2] if r[k] is not None else None) for k in ("pair", "trajectory")},
        used_for_selection=False, independent_confirmation=False, reserved_outcomes_opened=False)
    run.base.immutable(run.OUT/"contrast_attribution.json", out)
    reference = {k: run.av.interval(rows, lambda r: r[k][2] if r[k] is not None else None)
                 for k in ("reference_pair", "reference_trajectory")}
    reference["reference_qualification"] = run.av.interval(rows, lambda r: r["reference_qualification"])
    run.base.immutable(run.OUT/"frozen_reference_contrast.json", dict(source=study.artifact(path), result_source="fresh_run",
        views=len(rows), frozen_position_minus_soft=reference, used_for_selection=False,
        independent_confirmation=False, reserved_outcomes_opened=False))
    print(study.json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
