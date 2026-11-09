"""Attribute the frozen replay ablation on the original paired opportunity set."""

from collections import Counter
from pathlib import Path
import platform
import sys

if __name__ == "__main__" and sys.platform == "darwin" and platform.machine() != "arm64":
    raise SystemExit("Use native arm64 Python")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_history_rank as run

np, av, prev, study, base = run.np, run.av, run.prev, run.study, run.base


def direct_trajectory(after, before, cv, y, v):
    norm = lambda d: np.linalg.norm(d, axis=-1)
    mass = np.where(v, norm(cv-y), 0.).sum()
    if mass <= 0.:
        return None
    known = np.where(v, norm(before-y)-norm(after-y), 0.).sum()
    debit = np.where(v, 0., norm(after-before)).sum()
    return [float(100.*known/mass), float(100.*debit/mass), float(100.*(known-debit)/mass)]


def main():
    key, data, source, _, _, outer, old, _ = run.register()
    p = run.OUT/"summary.json"
    result = study.read(p)
    frozen = study.read(study.checked(result["freeze"]))
    if result["registration"] != key or frozen["registration"] != key or frozen["held_labels_used"]:
        raise ValueError("Require the sealed history comparison")
    truth = prev.load_targets(source, data, [n for n, d in data.items() if d["added"]])
    rows, counts = [], Counter()
    for fold, z in frozen["folds"].items():
        site, seed, cut = (z["identity"][k] for k in ("held", "seed", "cut"))
        with (np.load(study.checked(z["forecast"]), allow_pickle=False) as a,
              np.load(study.checked(outer["folds"][fold]["forecast"]), allow_pickle=False) as b,
              np.load(study.checked(old["folds"][fold]["forecast"]), allow_pickle=False) as c):
            for field in ("names", "offsets"):
                np.testing.assert_array_equal(a[field], b[field])
                np.testing.assert_array_equal(a[field], c[field])
            for i, (n, lo, hi) in enumerate(zip(a["names"], a["offsets"][:-1], a["offsets"][1:], strict=True)):
                d, (y, v) = data[n], truth[n]
                rr = run.eq.radii(d, seed)
                stats = {}
                if len(rr) == 2:
                    ref = c["additive"][lo:hi]
                    pool = run.eq.slates(d, rr, run.eq.expanded(d, b["raw_eq"][lo:hi]))
                    _, q, _ = prev.label_terms(d, pool, ref, y, v, cut)
                    eligible = run.eq.ceiling.eligible_additive(prev.metrics(ref, d, y, v, cut))
                    opportunity = bool(eligible and q.any())
                    stats.update(paired_opportunity=int(opportunity), queries=1)
                    for m in run.METHODS:
                        ix = a[f"top_{m}"][i]
                        ix = ix[ix >= 0]
                        good = bool(len(ix) and q[ix[0]])
                        top = bool(q[ix].any())
                        stats.update({f"{m}_paired_top5": int(opportunity and top),
                            f"{m}_paired_top1": int(opportunity and good)})
                counts.update(stats)
                rows.append(dict(clip=str(n), city=site, seed=seed, complete=bool(v.all()), **stats,
                    trajectory=direct_trajectory(a["history"][lo:hi], a["geometry"][lo:hi], d["cv"], y, v)))
    if (counts["paired_opportunity"] != result["counts"]["paired_ceiling"]
            or any(counts[f"{m}_paired_top1"] != result["counts"][f"{m}_both"] for m in run.METHODS)):
        raise ValueError("Matched opportunity audit did not replay")
    o = [r for r in rows if r.get("paired_opportunity")]
    out = dict(parent=study.artifact(p), code=study.artifact(Path(__file__)), totals=dict(counts),
        unique_eligible_scenarios=len({r["clip"] for r in o}),
        direct_trajectory_history_vs_geometry_pp={g: av.interval([r for r in rows if g == "all" or r["complete"]],
            lambda r: r["trajectory"]) for g in ("all", "complete")},
        term_order=["known", "missing_step_debit", "conservative_gain"], result_source="fresh_run",
        diagnostic_only=True, reserved_outcomes_opened=False, independent_confirmation=False)
    base.immutable(run.OUT/"attribution.json", out)
    print(study.json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
