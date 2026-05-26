import numpy as np

from src import stage42_source_level_ablation as s42an


def test_variant_masks_remove_expected_feature_groups():
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
    masks = s42an._variant_masks(names)
    assert masks["full"].sum() == len(names)
    assert not masks["no_history"][0]
    assert not masks["no_history"][2]
    assert not masks["no_goal_prototype"][3]
    assert not masks["no_baseline_family"][6]
    assert not masks["no_domain_expert"][9]
    assert masks["no_neighbor_interaction"][0]
    assert not masks["no_neighbor_interaction"][1]


def test_delta_identifies_positive_module_contribution():
    full = {"all_improvement": 0.20, "t50_improvement": 0.15, "t100_raw_frame_diagnostic_improvement": 0.10, "hard_failure_improvement": 0.18, "easy_degradation": 0.01}
    weak = {"all_improvement": 0.10, "t50_improvement": 0.14, "t100_raw_frame_diagnostic_improvement": 0.09, "hard_failure_improvement": 0.12, "easy_degradation": 0.00}
    d = s42an._delta(full, weak)
    assert d["all_improvement"] > s42an.MIN_MEANINGFUL_DELTA
    assert d["hard_failure_improvement"] > s42an.MIN_MEANINGFUL_DELTA


def test_gate_requires_multiple_positive_modules_and_no_leakage():
    metric = {"all_improvement": 0.2, "t50_improvement": 0.1, "hard_failure_improvement": 0.15, "easy_degradation": 0.0}
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "variants": {
            "full": {
                "protected": metric,
                "bootstrap": {"all": {"bootstrap_n": 1000}, "t50": {"bootstrap_n": 1000}},
            },
            "no_history": {},
            "no_neighbor_interaction": {},
            "no_goal_prototype": {},
            "no_baseline_family": {},
            "no_domain_expert": {},
        },
        "positive_components": ["history", "goal_prototype"],
        "positive_interaction_variants": [],
        "safe_switch_vs_ungated": {},
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
    gate = s42an._gate(result)
    assert gate["passed"] == gate["total"]


def test_gate_is_partial_for_only_one_independent_component():
    metric = {"all_improvement": 0.2, "t50_improvement": 0.1, "hard_failure_improvement": 0.15, "easy_degradation": 0.0}
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "variants": {
            "full": {
                "protected": metric,
                "bootstrap": {"all": {"bootstrap_n": 1000}, "t50": {"bootstrap_n": 1000}},
            },
            "no_history": {},
            "no_neighbor_interaction": {},
            "no_goal_prototype": {},
            "no_baseline_family": {},
            "no_domain_expert": {},
        },
        "positive_components": ["baseline_family_context"],
        "positive_interaction_variants": ["motion_goal_no_baseline_domain"],
        "safe_switch_vs_ungated": {},
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
    gate = s42an._gate(result)
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["at_least_two_independent_components_positive"] is False
