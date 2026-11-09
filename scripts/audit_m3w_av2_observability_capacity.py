"""Measure fixed-reference and relaxed-reference capacity on the frozen AV2 menu."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_observability as run

np, torch, study, base, eq, prev = run.np, run.torch, run.study, run.base, run.eq, run.prev


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, src, _, _, outer, control, _, _ = run.register()
    p = run.OUT/"summary.json"
    report = study.read(p)
    freeze = study.read(study.checked(report["freeze"]))
    if freeze["registration"] != key or freeze["held_labels_used"] or report["reserved_outcomes_opened"]:
        raise ValueError("Require frozen development readout")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    total, city, unique, availability = Counter(), {}, {}, Counter()
    for name, fold in freeze["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        tally = city.setdefault(held, Counter())
        with (np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as fixed):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(raw[k], fixed[k])
            for n, lo, hi in zip(raw["names"], raw["offsets"][:-1], raw["offsets"][1:], strict=True):
                d, (y, v) = data[n], truth[n]
                if len(rr := eq.radii(d, seed)) != 2:
                    continue
                if d["city"] != held:
                    raise ValueError("Invalid held lineage")
                pool = eq.slates(d, rr, eq.expanded(d, raw["raw_eq"][lo:hi]))
                a = fixed["additive"][lo:hi]
                v_edit = v[run.av.motion.peak(a-d["cv"]) > 0.]
                availability.update(grid=v_edit.size, known=int(v_edit.sum()),
                    edited=len(v_edit), completely_missing=int((~v_edit.any(1)).sum()))
                ref = prev.metrics(a, d, y, v, cut)
                ok = eq.ceiling.eligible_additive(ref)
                _, q, s = prev.label_terms(d, pool, a, y, v, cut)
                any_reference, any_pair = eq.ceiling.alternative_capacity(d, pool, y, v, cut)
                if (ok and not any_reference) or (ok and q.any() and not any_pair):
                    raise ValueError("Relaxation does not contain the observed witness")
                stats = dict(two_slots=1, fixed_support=int(ref["risk"]["completion_easy_risk_upper"] is not None),
                    any_support=int(s.any()), fixed_reference=int(ok), any_reference=int(any_reference),
                    fixed_pair=int(ok and q.any()), any_pair=int(any_pair))
                total.update(stats)
                tally.update(stats)
                for k, v in stats.items():
                    if v:
                        unique.setdefault(k, set()).add(str(n))
    if any(city[k]["fixed_reference"] != v["eligible"] or city[k]["two_slots"] != v["two_slots"]
           for k, v in report["by_city"].items()):
        raise ValueError("Fixed eligibility replay mismatch")
    out = dict(input=study.artifact(p), code=study.artifact(Path(__file__)), totals=dict(total),
        by_city={k: dict(v) for k, v in city.items()}, unique={k: len(v) for k, v in unique.items()},
        edited_actor_availability=dict(availability),
        result_source="fresh_run", hindsight_only=True, forecasts_or_thresholds_changed=False,
        independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(run.OUT/"reference_capacity.json", out)
    print(study.json.dumps({k: out[k] for k in ("totals", "by_city", "unique")}, indent=2))


if __name__ == "__main__":
    main()
