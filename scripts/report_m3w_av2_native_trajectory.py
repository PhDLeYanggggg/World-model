"""Separate incomplete-grid effects from the fixed-menu AV2 result."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_av2_native_trajectory as run

study, base = run.study, run.base


def summarize(result):
    support = result["sample_support"]
    views = result["per_scenario"]
    if Counter(r["clip"] for r in views) != {k: len(run.slate.SEEDS) for k, r in support.items() if r["roster"] >= 2}:
        raise ValueError("Missing or repeated scored views")
    if {(r["clip"], r["seed"]) for r in views} != {
            (k, s) for k, r in support.items() if r["roster"] >= 2 for s in run.slate.SEEDS}:
        raise ValueError("Scored AV2 views do not match the registered seeds")
    if any(r["city"] != support[r["clip"]]["city"] for r in views):
        raise ValueError("City cluster changed")
    complete = [r for r in views if support[r["clip"]]["roster"] == support[r["clip"]]["complete"]]
    qualified = 0
    for r in views:
        a, j = (r["arms"]["all"][arm] for arm in ("additive", "relative_joint"))
        if (a["risk"]["selected"] != j["risk"]["selected"] or
                (r["pair"]["all"] is None and support[r["clip"]]["roster"] == support[r["clip"]]["complete"])):
            raise ValueError("Unpaired arms or unscored complete roster")
        # On complete rosters the available and full-grid pair reductions coincide.
        if r in complete:
            run.np.testing.assert_allclose(*r["pair"]["all"], atol=1e-10)
            qualified += bool(j["risk"]["supported"] and min(j["gain"]) > 0.
                              and j["ade_complete"] <= a["ade_complete"]+1e-10 and r["pair"]["all"][1] > 1e-10)
    types = Counter()
    for r in support.values():
        types.update(r.get("types", {}))
    pair = lambda rows: run.interval(rows, lambda r: r["pair"]["all"])
    return dict(
        support=dict(scenarios=len(support), cities=dict(Counter(r["city"] for r in support.values())),
                     unique_roster_rows=sum(r["roster"] for r in support.values() if r["roster"] >= 2),
                     complete_rows=sum(r.get("complete", 0) for r in support.values()),
                     endpoint_rows=sum(r.get("endpoint", 0) for r in support.values()),
                     labelled_steps=sum(r.get("grid", 0) for r in support.values()),
                     complete_scenarios=len({r["clip"] for r in complete}), roster_types=dict(types),
                     pedestrian_pair_scenarios=len({r["clip"] for r in views if r["pair"].get("pedestrian") is not None})),
        joint_comparison=dict(
            all_pair_native=pair(views), complete_pair_native=pair(complete),
            complete_two_slot_pair_native=pair([r for r in complete if r["arms"]["all"]["anchor"]["risk"]["selected"] == 2]),
            per_seed_pair_native={s: pair([r for r in views if r["seed"] == s]) for s in run.slate.SEEDS},
            per_city_pair_native={c: pair([r for r in views if r["city"] == c])["site_mean"]
                                  for c in sorted({r["city"] for r in views})},
            complete_qualified_joint_wins=qualified, ceiling=result["exact_complete_two_slot_ceiling"]),
        all_arms=result["totals"]["all"]["arms"], pedestrian=result["totals"]["pedestrian"])


def main():
    path = run.OUT/"readout.json"
    result = study.read(path)
    if (result["registration"] != base.sha(run.OUT/"registration.json") or result["independent_confirmation"]
            or result["reserved_outcomes_opened"] or not result["no_training_or_threshold_search"]):
        raise ValueError("Require the registered development-only comparison")
    freeze = study.read(study.checked(result["freeze"]))
    if freeze["registration"] != result["registration"] or freeze["future_rows_used"]:
        raise ValueError("Frozen forecast input does not match")
    base.immutable(run.OUT/"summary.json", dict(source="fresh_run_aggregation",
        readout=study.artifact(path), raw_manifest=study.artifact(run.SOURCE/"manifest.json"),
        independent_confirmation=False, **summarize(result)))


if __name__ == "__main__":
    main()
