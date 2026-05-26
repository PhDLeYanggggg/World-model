import numpy as np

from src import stage42_source_level_baseline_family_mechanism as s42au


def test_variant_masks_split_baseline_family_components():
    names = [
        "floor_rel_x",
        "safe_baseline_rel_0",
        "family_baseline_rel_0",
        "horizon_50",
        "domain_UCY",
        "history_scalar_0",
    ]
    masks = s42au._variant_masks(names)
    assert masks["horizon_domain_control"].tolist() == [False, False, False, True, True, False]
    assert masks["floor_rel_only"].tolist() == [True, False, False, True, True, False]
    assert masks["safe_baseline_rel_only"].tolist() == [False, True, False, True, True, False]
    assert masks["family_baseline_rel_only"].tolist() == [False, False, True, True, True, False]
    assert masks["baseline_family_all"].tolist() == [True, True, True, True, True, False]


def test_core_delta_supported_uses_core_metrics():
    assert s42au._core_delta_supported({"all_improvement": 0.02, "t50_improvement": -1.0, "hard_failure_improvement": -1.0})
    assert s42au._core_delta_supported({"all_improvement": 0.0, "t50_improvement": 0.02, "hard_failure_improvement": 0.0})
    assert not s42au._core_delta_supported({"all_improvement": 0.005, "t50_improvement": 0.0, "hard_failure_improvement": 0.0})


def test_best_single_family_selects_highest_all_improvement():
    variants = {
        "floor_rel_only": {"protected": {"all_improvement": 0.1}},
        "safe_baseline_rel_only": {"protected": {"all_improvement": 0.3}},
        "family_baseline_rel_only": {"protected": {"all_improvement": 0.2}},
    }
    name, metric = s42au._best_single_family(variants, "protected")
    assert name == "safe_baseline_rel_only"
    assert metric["all_improvement"] == 0.3


def test_gate_accepts_supported_mechanism_without_metric_claim():
    metric = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.0,
        "easy_degradation": 0.0,
    }
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "variants": {
            "baseline_family_all": {"protected": metric, "ungated": metric},
            "horizon_domain_control": {},
            "floor_rel_only": {},
            "safe_baseline_rel_only": {},
            "family_baseline_rel_only": {},
            "floor_plus_safe": {},
            "floor_plus_family": {},
            "safe_plus_family": {},
        },
        "summary": {"mechanism_verdict": "baseline_family_rollout_context_supported_as_dominant_mechanism"},
        "protected_family_increment": {"baseline_family_all_minus_best_single": {}},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42au._gate(result)
    assert gate["passed"] == gate["total"]
