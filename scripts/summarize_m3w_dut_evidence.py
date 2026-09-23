"""Describe every frozen DUT family, without selecting an external winner."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_recording_lineage import sha256
from scripts.run_m3w_native_forecast import immutable_json

PUBLIC = ROOT / "outputs/publication_readiness_2026_09/dut_frozen_readout_v1"


def summarize(m):
    groups = {}
    for name, arms in m["fixed_three_seed_groups"].items():
        groups[name] = {arm: arms[arm] for arm in ("strict", "uncontrolled", "half_independent", "half_unary", "half_joint")}
    details = {}
    for name, v in m["views"].items():
        arms = v["arms"]
        strict = arms["strict"]
        details[name] = dict(
            strict_improvement=strict["subsets"]["all"]["improvement_percent"],
            strict_switches=sum(s["switches"] for s in strict["sites"].values()),
            strict_switch_rate=sum(s["switches"] for s in strict["sites"].values()) / m["coverage"]["targets"],
            strict_easy_harmed=sum(s["subsets"]["easy"]["harmed"] for s in strict["sites"].values()),
            strict_full_population_gain_interval=strict["full_population_gain_interval"],
            half_joint_minus_unary_gain_pp=arms["half_joint"]["subsets"]["all"]["improvement_percent"]
                - arms["half_unary"]["subsets"]["all"]["improvement_percent"],
            half_joint_changed_query_view_instances=v["joint_different_queries"],
            half_unmatched=v["half_unmatched"], rejected_outputs=v["rejected_outputs"])
    first = next(iter(m["views"].values()))["arms"]["strict"]
    easy_n = sum(s["subsets"]["easy"]["n"] for s in first["sites"].values())
    return dict(result_source="fresh_run_descriptive_summary_of_hash_bound_readout",
        fixed_seed_groups=groups, views=details,
        easy_complete_targets=easy_n,
        easy_fraction_of_complete_targets=easy_n / m["coverage"]["complete"],
        query_view_instances=m["coverage"]["queries"] * len(m["views"]),
        half_joint_changed_query_view_instances=sum(d["half_joint_changed_query_view_instances"] for d in details.values()),
        all_strict_full_population_gain_lower_bounds_positive=all(d["strict_full_population_gain_interval"][0] > 0 for d in details.values()),
        all_strict_empirical_easy_harmed_zero=all(d["strict_easy_harmed"] == 0 for d in details.values()),
        observed_aggregate_leader_is_not_selected_for_deployment="damped_velocity_005",
        damping005=m["baseline_controls"]["damped_velocity_005"]["subsets"],
        selected_model=None, population_safety_certified=False, independent_confirmation=False,
        joint_contribution_supported=False, new_deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    metric_path = PUBLIC / "metrics.json"
    m = json.loads(metric_path.read_text())
    if m["analysis_sha256"] != sha256(PUBLIC / "analysis.json"):
        raise ValueError("Readout binding changed")
    s = summarize(m)
    s.update(metrics_sha256=sha256(metric_path), code_sha256=sha256(Path(__file__)))
    immutable_json(PUBLIC / "evidence_summary.json", s)
    print(json.dumps({k: v for k, v in s.items() if k not in ("fixed_seed_groups", "views", "damping005")}, indent=2))


if __name__ == "__main__":
    main()
