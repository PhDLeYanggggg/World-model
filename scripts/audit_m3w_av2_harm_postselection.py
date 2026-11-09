"""Paired coverage readout before and after using the same upper score to choose."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_harm_quantile as run

np, torch, study, base, av, prev, old, eq = (
    run.np, run.torch, run.study, run.base, run.av, run.prev, run.old, run.eq)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, src, _, _, outer, _, _, _ = run.register()
    path = run.OUT/"summary.json"
    result = study.read(path)
    seal = study.read(study.checked(result["freeze"]))
    if seal["registration"] != key or seal["held_labels_used"] or result["reserved_outcomes_opened"]:
        raise ValueError("Require the frozen development readout")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    rows = []
    for name, fold in seal["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        with (np.load(study.checked(fold["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw):
            for k in ("names", "offsets"):
                np.testing.assert_array_equal(z[k], raw[k])
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                if z["ix_prior"][i] < 0:
                    continue
                d = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi]))
                if d["city"] != held:
                    raise ValueError("Wrong held lineage")
                menu = old.action.menu(d, seed, cut)
                truth_terms = np.asarray([run.audit.actual(d, menu, *truth[n], cut, int(z[f"ix_{m}"][i]))
                                          for m in ("prior", "upper")])
                prediction = np.asarray([z[f"harm_{m}"][i, 1] for m in ("prior", "upper")])
                covers = (truth_terms <= prediction).astype(int)
                changed = int(not np.array_equal(z["prior"][lo:hi], z["upper"][lo:hi]))
                rows.append(dict(city=held, clip=str(n), seed=seed, changed=changed, coverage=covers.tolist(),
                    delta=(covers[1]-covers[0]).tolist(), residual_delta=((truth_terms-prediction)[1]-(truth_terms-prediction)[0]).tolist()))
    if len(rows) != result["counts"]["two_slots"]:
        raise ValueError("Missing two-slot views")
    coverage = np.mean([r["coverage"] for r in rows], axis=0)
    for k, value in zip(("prior", "upper"), coverage, strict=True):
        np.testing.assert_allclose(value, result["harm"][k]["upper_coverage"], atol=1e-15, rtol=0.)
    payload = dict(input=study.artifact(path), code=study.artifact(Path(__file__)), result_source="fresh_run",
        two_slot_seed_views=len(rows), two_slot_scenarios=len({r["clip"] for r in rows}),
        changed_paths=sum(r["changed"] for r in rows), terms=result["harm_terms"], pooled_coverage=coverage.tolist(),
        paired_delta=av.interval(rows, lambda r: r["delta"]),
        paired_residual_delta=av.interval(rows, lambda r: r["residual_delta"]),
        by_city={s: dict(views=sum(r["city"] == s for r in rows),
                        coverage_delta=np.mean([r["delta"] for r in rows if r["city"] == s], axis=0).tolist())
                 for s in sorted({r["city"] for r in rows})},
        contrast="same upper-risk predictor and scenes, used on frozen versus newly selected options",
        forecasts_or_thresholds_changed=False, independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(run.OUT/"postselection.json", payload)
    print(study.json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
