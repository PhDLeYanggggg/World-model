"""Attribute forced matched choices without reopening the release decision."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_relative_margins as run

np, torch, study, base, av, prev, eq, cost = (
    run.np, run.torch, run.study, run.base, run.av, run.prev, run.eq, run.cost)
MODES = ("prior", *run.MODES)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    key, data, src, _, _, outer, control, _, _, previous, _ = run.register()
    source = run.OUT/"summary.json"
    report = study.read(source)
    seal = study.read(study.checked(report["freeze"]))
    if seal["registration"] != key or seal["held_labels_used"] or report["reserved_outcomes_opened"]:
        raise ValueError("Require a sealed and scored development result")
    truth = prev.load_targets(src, data, [n for n, d in data.items() if d["added"]])
    subsets = ("two_slots", "anchor_eligible", "paired_available", "complete")
    counts, rows, by_city = {k: {m: Counter() for m in MODES} for k in subsets}, [], {}
    for name, fold in seal["folds"].items():
        held, seed, cut = (fold["identity"][k] for k in ("held", "seed", "cut"))
        action = run.pair.prior.load_action(control["folds"][name])
        city = by_city.setdefault(held, Counter())
        with (np.load(study.checked(fold["forecast"]), allow_pickle=False) as z,
              np.load(study.checked(control["folds"][name]["forecast"]), allow_pickle=False) as c,
              np.load(study.checked(previous["folds"][name]["forecast"]), allow_pickle=False) as old,
              np.load(study.checked(outer["folds"][name]["forecast"]), allow_pickle=False) as raw):
            for k in ("names", "offsets"):
                for bank in (c, old, raw):
                    np.testing.assert_array_equal(z[k], bank[k])
            for i, (n, lo, hi) in enumerate(zip(z["names"], z["offsets"][:-1], z["offsets"][1:], strict=True)):
                d, (y, v) = dict(data[n], bank=eq.expanded(data[n], raw["raw_eq"][lo:hi])), truth[n]
                if (s := cost.slate(d, seed, cut, action)) is None:
                    continue
                paths, a = s[:2]
                np.testing.assert_array_equal(a, c["additive"][lo:hi])
                missing = not v.any(1).all()
                t = run.targets(d, s, y, v, cut)
                anchor = bool(t[0, 1])
                available = bool(t[:, 0].any())
                scope = dict(two_slots=True, anchor_eligible=anchor, paired_available=available, complete=bool(v.all()))
                city.update(two_slots=1, missing_actor=int(missing), missing_actor_candidates=len(paths)*missing,
                            anchor_eligible=int(anchor), paired_available=int(available))
                picks = dict(prior=old["top_direct"][i, 0], **{m: z[f"top_{m}"][i, 0] for m in run.MODES})
                row = dict(city=held, clip=str(n), seed=seed, complete=scope["complete"])
                for mode, j in picks.items():
                    p = paths[j]
                    met = prev.metrics(p, d, y, v, cut)
                    traj = cost.audit.direct_trajectory(p, a, d["cv"], y, v)
                    relative = av.frontier.pair_reduction(p[None], a, y, v)[0, 1]
                    risk = met["risk"]
                    fields = ("completion_all_risk_upper", "completion_easy_risk_upper", "known_easy_net_degradation")
                    checks = dict(total=1, anchor_ineligible=int(not anchor),
                        all_positive_harm=int(risk[fields[0]] is None or risk[fields[0]] > .02),
                        easy_unsupported=int(risk[fields[1]] is None or risk[fields[2]] is None),
                        easy_positive_harm=int(risk[fields[1]] is not None and risk[fields[1]] > .02),
                        easy_signed_harm=int(risk[fields[2]] is not None and risk[fields[2]] > .02),
                        dual_floor=int(met["gain"] is None or min(met["gain"]) <= 0.),
                        relative_trajectory=int(traj is None or traj[2] < -1e-8),
                        relative_pair=int(relative <= 1e-10))
                    passed = not any(v for k, v in checks.items() if k != "total")
                    if passed != bool(t[j, 0]):
                        raise ValueError("Forced-forecast risk decomposition disagrees with scalar checks")
                    checks["matched"] = int(passed)
                    for k in subsets:
                        if scope[k]:
                            counts[k][mode].update(checks)
                    row[mode] = [traj[2], float(relative)] if traj is not None else None
                rows.append(row)
    for m in MODES:
        if counts["two_slots"][m]["matched"] != report["counts"][f"{m}_top1"]:
            raise ValueError("Matched top-1 count changed")
    out = dict(result_source="fresh_run", source=study.artifact(source), code=study.artifact(Path(__file__)),
        counts={k: {m: dict(v) for m, v in x.items()} for k, x in counts.items()},
        missingness_by_city={k: dict(v) for k, v in by_city.items()},
        forced_gain_vs_fixed={k: {m: av.interval([r for r in rows if k == "all" or r["complete"]], lambda r: r[m])
                                  for m in MODES} for k in ("all", "complete")},
        gain_order=["trajectory_lower_pp", "pair_lower_native"], original_2pct=True,
        changed_choices=False, reserved_outcomes_opened=False, independent_confirmation=False)
    base.immutable(run.OUT/"attribution.json", out)
    print(study.json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
