"""Recompute the direct paired effect of adding the interaction ranker."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_conditional_pair_gain as run

np, study, base = run.np, run.study, run.base
frontier, effects = run.target.frontier, run.pair.run.turn.effects


def direct(p, d):
    q = np.unique(d["q"], return_inverse=True)[1]
    v, y = d["v"], d["y"]
    before, after = p["feasibility"], p["pair_gain"]
    if not (before != p["cv"]).any():
        return None
    den = frontier.denominators(q, v)
    if not den.all():
        return None
    change = frontier.aggregate_pair(after, before, y, v, q, den)
    w = 1./np.bincount(q)[q]
    mass = float(w@np.where(v, np.linalg.norm(p["cv"]-y, axis=-1), 0.).sum(1))
    traj = 100.*float(w@run.target.lower(before, after, y, v))/mass if mass > 0. else None
    return dict(pair=change.tolist(), trajectory_lower_pp=traj)


def main():
    report_path = run.OUT/"summary.json"
    result = study.read(report_path)
    frozen = study.read(study.checked(result["freeze"]))
    key, cohorts, _ = run.register()
    if frozen["registration"] != key or result["registration"] != key or frozen["held_outcomes_used"]:
        raise ValueError("Require the frozen comparison")
    data = run.pair.run.turn.exp.loader(cohorts)
    rows, summaries = [], {}
    for site in run.SITES:
        src = data[site]
        local = []
        for seed in run.SEEDS:
            item = frozen["folds"][f"{site}_{seed}"]
            p = run.rank.checked_npz(item["held"]["forecast"])
            for r in np.unique(src["rec"]):
                ids = src["rec"] == r
                contrast = direct({k: p[k][ids] for k in ("feasibility", "pair_gain", "cv")},
                                  {k: src[k][ids] for k in ("q", "v", "y")})
                if contrast is not None:
                    local.append(dict(record=f"{site}:{r}", seed=seed, **contrast))
        summaries[site] = dict(pair=effects.site_means([(r["record"], r["pair"]) for r in local]),
            trajectory=effects.site_means([(r["record"], r["trajectory_lower_pp"])
                                           for r in local if r["trajectory_lower_pp"] is not None]))
        rows.extend(local)
    cal = {m: {k: sum(r[m][k] for r in result["thresholds"].values())
               for k in ("nonempty_breakpoints", "feasible_breakpoints")} for m in run.MODES}
    out = dict(summary=study.artifact(report_path), result_source="fresh_run",
        site_means=summaries, estimable_views=len(rows), unique_recordings=len({r["record"] for r in rows}),
        direct_pair_native=effects.interval([r["pair"] for r in summaries.values()]),
        direct_trajectory_lower_pp=effects.interval([r["trajectory"] for r in summaries.values()]),
        calibration_breakpoints=cal, independent_confirmation=False, reserved_outcomes_opened=False)
    base.immutable(run.OUT/"direct_contrast.json", out)
    print(study.json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
