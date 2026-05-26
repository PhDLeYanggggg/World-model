import numpy as np

from src import stage42_goal_scene_gated_expert as s42cj


def test_variant_masks_include_goal_scene_and_baseline_controls():
    names = [
        "history_scalar_0",
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
    masks = s42cj._variant_masks(names)
    assert masks["baseline_family_control"][5]
    assert masks["baseline_family_control"][8]
    assert masks["baseline_family_control"][9]
    assert not masks["baseline_family_control"][2]
    assert masks["goal_scene_only_control"][2]
    assert masks["goal_scene_only_control"][4]
    assert not masks["goal_scene_only_control"][5]
    assert masks["baseline_plus_goal_scene"][2]
    assert masks["baseline_plus_goal_scene"][6]
    assert masks["baseline_plus_motion_goal_context"][0]


def test_select_deployment_variant_requires_validation_margin_and_easy_safety():
    baseline = {
        "variant": "baseline_family_control",
        "validation_selection": {"selected_score": 1.0, "candidates": [{"score": 1.0, "val_metric": {"easy_degradation": 0.0}}]},
    }
    candidate = {
        "variant": "baseline_plus_goal_scene",
        "validation_selection": {"selected_score": 1.02, "candidates": [{"score": 1.02, "val_metric": {"easy_degradation": 0.01}}]},
    }
    unsafe = {
        "variant": "baseline_plus_motion_goal_context",
        "validation_selection": {"selected_score": 1.08, "candidates": [{"score": 1.08, "val_metric": {"easy_degradation": 0.03}}]},
    }
    selected = s42cj._select_deployment_variant(
        {
            "baseline_family_control": baseline,
            "baseline_plus_goal_scene": candidate,
            "baseline_plus_motion_goal_context": unsafe,
        }
    )
    assert selected["selected_variant"] == "baseline_plus_goal_scene"
    assert selected["test_threshold_tuning"] is False


def test_select_deployment_variant_falls_back_on_low_margin():
    baseline = {
        "variant": "baseline_family_control",
        "validation_selection": {"selected_score": 1.0, "candidates": [{"score": 1.0, "val_metric": {"easy_degradation": 0.0}}]},
    }
    low_margin = {
        "variant": "baseline_plus_goal_scene",
        "validation_selection": {"selected_score": 1.005, "candidates": [{"score": 1.005, "val_metric": {"easy_degradation": 0.0}}]},
    }
    motion = {
        "variant": "baseline_plus_motion_goal_context",
        "validation_selection": {"selected_score": 0.99, "candidates": [{"score": 0.99, "val_metric": {"easy_degradation": 0.0}}]},
    }
    selected = s42cj._select_deployment_variant(
        {
            "baseline_family_control": baseline,
            "baseline_plus_goal_scene": low_margin,
            "baseline_plus_motion_goal_context": motion,
        }
    )
    assert selected["selected_variant"] == "baseline_family_control"


def test_goal_scene_rescue_success_requires_switch_and_core_gain():
    baseline = {
        "variant": "baseline_family_control",
        "protected": {
            "all_improvement": 0.20,
            "t50_improvement": 0.10,
            "t100_raw_frame_diagnostic_improvement": 0.0,
            "hard_failure_improvement": 0.12,
            "easy_degradation": 0.0,
            "switch_rate": 0.1,
            "harm_over_fallback": -0.1,
        },
    }
    selected = {
        "variant": "baseline_plus_goal_scene",
        "protected": {
            "all_improvement": 0.20,
            "t50_improvement": 0.115,
            "t100_raw_frame_diagnostic_improvement": 0.0,
            "hard_failure_improvement": 0.12,
            "easy_degradation": 0.01,
            "switch_rate": 0.1,
            "harm_over_fallback": -0.1,
        },
    }
    assert s42cj._goal_scene_rescue_success(selected, baseline)
    unsafe = dict(selected)
    unsafe["protected"] = dict(selected["protected"])
    unsafe["protected"]["easy_degradation"] = 0.03
    assert not s42cj._goal_scene_rescue_success(unsafe, baseline)


def test_gate_allows_diagnostic_pass_without_goal_scene_overclaim():
    metric = {
        "all_improvement": 0.2,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.12,
        "easy_degradation": 0.0,
    }
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "variants": {
            "baseline_family_control": {"protected": metric},
            "baseline_plus_goal_scene": {"protected": metric},
            "baseline_plus_motion_goal_context": {"protected": metric},
        },
        "validation_only_selection": {"selected_variant": "baseline_family_control", "baseline_variant": "baseline_family_control", "test_threshold_tuning": False},
        "goal_scene_rescue_success": False,
        "claim_boundary": {"goal_scene_main_claim_allowed": False, "metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
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
    }
    gate = s42cj._gate(result)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_cj_goal_scene_gated_expert_pass_diagnostic_no_overclaim"
