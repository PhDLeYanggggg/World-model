"""Paired same-count risk and gain audit on the unchanged held forecasts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_projection_risk as run

np, study = run.np, run.study
FIELDS = ("qualified", "risk_fail", "easy_over", "floor_fail")


def weights(scores, budget):
    k = min(len(scores), budget)
    if not k:
        return np.zeros_like(scores, dtype=float)
    cutoff = np.partition(scores, len(scores)-k)[-k]
    above = scores > cutoff
    ties = scores == cutoff
    return above.astype(float)+ties*((k-above.sum())/ties.sum())


def counts(rows, arm, budget):
    w = weights(np.array([r["scores"][arm] for r in rows]), budget)
    values = np.array([(r["result"]["eligible"], not r["result"]["risk"], r["result"]["easy_over"],
                       r["result"]["gain"] is None or min(r["result"]["gain"]) <= 0.) for r in rows])
    return dict(zip(FIELDS, map(float, w@values), strict=True))


def main():
    key, _, _, _, _, _, _, _, parent, _, _ = run.register()
    path = run.OUT/"summary.json"
    s = study.read(path)
    f = study.read(study.checked(s["freeze"]))
    if f["registration"] != key or f["held_labels_used"] or s["reserved_outcomes_opened"]:
        raise ValueError("Require frozen matched-score study")
    training_models, direction_roles, options = set(), 0, 0
    cells = {k: [] for k in (1, 5, 10)}
    for name, ref in f["folds"].items():
        z = study.read(study.checked(ref))
        for c, l in z["lineage"].items():
            excluded = {z["identity"]["held"], z["identity"]["calibration"], c}
            if (set(l["training"]) != set(z["identity"]["donors"])-{c}
                    or excluded & set(l["training"]) or l["direction"]["identity"]["training"] != l["training"]):
                raise ValueError("Projection training exposure")
            direction_roles += 1
            training_models.add(l["direction"]["checkpoint"]["sha256"])
        options += z["fit"]["rows"]
        rows = list(run.prior.role_rows(z, parent["folds"][name], name, "held"))
        for k, d in cells.items():
            if not rows:
                continue
            d.append(dict(city=z["identity"]["held"], clip=f"same_count_{k}", n=min(k, len(rows)),
                          arms={a: counts(rows, a, k) for a in run.model.ARMS}))
    out = {}
    for k, rows in cells.items():
        total = sum(r["n"] for r in rows)
        totals = {a: {m: sum(r["arms"][a][m] for r in rows) for m in FIELDS} for a in run.model.ARMS}
        if any(abs(totals[a]["qualified"]-s["matched_counts"][a][f"top{k}"]) > 1e-8 for a in run.model.ARMS):
            raise ValueError("Fixed-count qualification did not replay")
        paired = {a: {m: run.av.interval(rows, lambda r: 100.*(r["arms"]["gain_risk"][m]-r["arms"][a][m])/r["n"])
                      for m in FIELDS} for a in ("gain_only", "transferred_cost", "risk_replay")}
        out[k] = dict(same_count=total, expected_counts=totals, gain_risk_minus_comparator_pp=paired)
    run.base.immutable(run.OUT/"matched_audit.json", dict(source=study.artifact(path), result_source="fresh_run",
        used_for_selection=False, independent_confirmation=False, reserved_outcomes_opened=False,
        unique_inner_direction_fits=len(training_models), direction_roles=direction_roles, training_options=options, budgets=out))
    print(study.json.dumps({k: v["expected_counts"] for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
