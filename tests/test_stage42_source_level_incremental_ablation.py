import numpy as np

from src import stage42_source_level_incremental_ablation as s42ao


def test_incremental_variant_masks_include_expected_controls():
    names = [
        "history_scalar_0",
        "history_scalar_1",
        "history_tail_0",
        "prototype_0",
        "prototype_entropy",
        "goal_ambiguity",
        "safe_baseline_rel_0",
        "family_baseline_rel_0",
        "floor_rel_x",
        "domain_UCY",
        "horizon_50",
    ]
    masks = s42ao._incremental_variant_masks(names)
    assert masks["full"].sum() == len(names)
    assert masks["horizon_domain_only"][9]
    assert masks["horizon_domain_only"][10]
    assert not masks["horizon_domain_only"][0]
    assert masks["baseline_family_only"][6]
    assert masks["baseline_family_only"][9]
    assert masks["history_only"][0]
    assert masks["goal_only"][3]
    assert masks["baseline_plus_goal"][7]
    assert masks["baseline_plus_goal"][3]


def test_metric_delta_and_positive_core_delta():
    stronger = {
        "all_improvement": 0.20,
        "t50_improvement": 0.12,
        "t100_raw_frame_diagnostic_improvement": 0.02,
        "hard_failure_improvement": 0.16,
        "easy_degradation": 0.01,
        "switch_rate": 0.2,
        "harm_over_fallback": -0.1,
    }
    weaker = {
        "all_improvement": 0.18,
        "t50_improvement": 0.11,
        "t100_raw_frame_diagnostic_improvement": 0.02,
        "hard_failure_improvement": 0.15,
        "easy_degradation": 0.01,
        "switch_rate": 0.2,
        "harm_over_fallback": -0.08,
    }
    delta = s42ao._metric_delta(stronger, weaker)
    assert delta["all_improvement"] > s42ao.MIN_INCREMENTAL_DELTA
    assert s42ao._positive_core_delta(delta)


def test_standalone_positive_requires_easy_safety():
    metric = {
        "all_improvement": 0.04,
        "t50_improvement": 0.0,
        "hard_failure_improvement": 0.0,
        "easy_degradation": 0.01,
    }
    assert s42ao._standalone_positive(metric)
    unsafe = dict(metric)
    unsafe["easy_degradation"] = 0.03
    assert not s42ao._standalone_positive(unsafe)


def test_gate_records_partial_when_incremental_context_missing():
    metric = {"all_improvement": 0.2, "t50_improvement": 0.1, "hard_failure_improvement": 0.15, "easy_degradation": 0.0}
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "variants": {
            "full": {
                "protected": metric,
                "bootstrap": {"all": {"bootstrap_n": 1000}, "t50": {"bootstrap_n": 1000}},
            },
            "baseline_family_only": {"protected": {"all_improvement": 0.1}},
            **{f"v{i}": {} for i in range(9)},
        },
        "positive_standalone_context_variants": ["history_only"],
        "positive_incremental_context_variants": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42ao._gate(result)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["incremental_context_signal_found"] is False
