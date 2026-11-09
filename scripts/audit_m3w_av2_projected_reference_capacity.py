"""Separate direction from recipient opportunity on scored calibration views."""

from collections import Counter, defaultdict
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_projected_reference as run
from scripts.audit_m3w_av2_steered_reference_ceiling import best_direction

np, study = run.np, run.study


def main():
    key, data, src, _, _, outer, control, _, _, _ = run.register()
    p = run.OUT/"summary.json"
    result = study.read(p)
    seal = study.read(study.checked(result["freeze"]))
    if (seal["registration"] != key or not result["calibration_only"] or result["held_labels_opened"]
            or result["reserved_outcomes_opened"] or seal["calibration_labels_used"]):
        raise ValueError("Require already scored calibration-only forecasts")
    cities = defaultdict(Counter)
    for name, ref in seal["folds"].items():
        f = study.read(study.checked(ref))
        parent = study.read(study.checked(f["identity"]["calibration_parent"]))
        r = study.read(run.LOCAL/f"{name}.score.json")
        if r["freeze"] != ref:
            raise ValueError("Wrong scoring receipt")
        city, seed, cut = (f["identity"][k] for k in ("calibration", "seed", "cut"))
        names = [z["clip"] for z in f["rows"]]
        if city in [f["identity"]["held"], *f["identity"]["donors"]] or any(data[n]["city"] != city for n in names):
            raise ValueError("Calibration-only target access")
        truth = run.prev.load_targets(src, data, names)
        count = cities[city]
        for (n, d, bank, opts, _, digest), old, z, scored in zip(
                run.prior.view_paths(data, parent["names"], control["folds"][name], outer["folds"][name], seed, cut),
                parent["rows"], f["rows"], r["rows"], strict=True):
            ix = old["choices"][1]
            if not n == z["clip"] == scored["clip"] or z["hash"] != digest or opts[ix].tolist() != z["edited"]:
                raise ValueError("Wrong fixed comparison")
            y, v = truth[n]
            cv, ols = d["cv"], d["ols4"]
            free = best_direction(cv[:, None], ols[:, None], bank, y[:, None], v[:, None], np.pi)
            ok, support = run.bank.model.eligibility(cv, ols, free, y, v, cut[0], opts)
            if any(scored["arms"][k]["eligible"] for k in run.ARMS) and not ok[ix]:
                raise ValueError("A causal proposal exceeds its fixed-recipient upper bound")
            count["views"] += 1
            count["fixed_recipient_support"] += int(support[ix])
            count["fixed_recipient_free_direction"] += int(ok[ix])
            count["any_recipient_free_direction"] += int(ok.any())
            count["fixed_recipient_free_direction_complete"] += int(ok[ix] and v.all())
            for k in run.ARMS:
                count[f"qualified:{k}"] += int(scored["arms"][k]["eligible"])
    total = sum(cities.values(), Counter())
    if total["views"] != result["views"] or any(total[f"qualified:{k}"] != result["arms"][k]["eligible"] for k in run.ARMS):
        raise ValueError("Frozen score did not replay")
    run.base.immutable(run.OUT/"direction_capacity.json", dict(source=study.artifact(p), result_source="fresh_run",
        calibration_only=True, hindsight_only=True, used_for_selection=False, reserved_outcomes_opened=False,
        held_labels_opened=False, counts=total, by_city=cities))
    print(study.json.dumps(total, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
