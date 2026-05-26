import numpy as np

from src import stage42_neighbor_interaction_gated_expert as s42ck


def test_or_mask_combines_boolean_masks():
    a = np.array([True, False, False])
    b = np.array([False, True, False])
    out = s42ck._or_mask(a, b)
    assert out.tolist() == [True, True, False]


def test_select_deployment_variant_picks_safe_validation_winner():
    baseline = {
        "variant": "baseline_family_control",
        "validation_selection": {"selected_score": 1.0, "candidates": [{"score": 1.0, "val_metric": {"easy_degradation": 0.0}}]},
    }
    scalar = {
        "variant": "baseline_plus_scalar_neighbor",
        "validation_selection": {"selected_score": 0.9, "candidates": [{"score": 0.9, "val_metric": {"easy_degradation": 0.0}}]},
    }
    graph = {
        "variant": "baseline_plus_knn_graph",
        "validation_selection": {"selected_score": 1.03, "candidates": [{"score": 1.03, "val_metric": {"easy_degradation": 0.01}}]},
    }
    graph_goal = {
        "variant": "baseline_plus_graph_goal",
        "validation_selection": {"selected_score": 1.04, "candidates": [{"score": 1.04, "val_metric": {"easy_degradation": 0.03}}]},
    }
    graph_history = {
        "variant": "baseline_plus_graph_history_scalar",
        "validation_selection": {"selected_score": 0.95, "candidates": [{"score": 0.95, "val_metric": {"easy_degradation": 0.0}}]},
    }
    selected = s42ck._select_deployment_variant(
        {
            "baseline_family_control": baseline,
            "baseline_plus_scalar_neighbor": scalar,
            "baseline_plus_knn_graph": graph,
            "baseline_plus_graph_goal": graph_goal,
            "baseline_plus_graph_history_scalar": graph_history,
        }
    )
    assert selected["selected_variant"] == "baseline_plus_knn_graph"
    assert selected["test_threshold_tuning"] is False


def test_select_deployment_variant_falls_back_when_no_safe_margin():
    baseline = {
        "variant": "baseline_family_control",
        "validation_selection": {"selected_score": 1.0, "candidates": [{"score": 1.0, "val_metric": {"easy_degradation": 0.0}}]},
    }
    weak = {
        "variant": "baseline_plus_scalar_neighbor",
        "validation_selection": {"selected_score": 1.005, "candidates": [{"score": 1.005, "val_metric": {"easy_degradation": 0.0}}]},
    }
    unsafe = {
        "variant": "baseline_plus_knn_graph",
        "validation_selection": {"selected_score": 1.05, "candidates": [{"score": 1.05, "val_metric": {"easy_degradation": 0.04}}]},
    }
    filler = {
        "variant": "filler",
        "validation_selection": {"selected_score": 0.5, "candidates": [{"score": 0.5, "val_metric": {"easy_degradation": 0.0}}]},
    }
    selected = s42ck._select_deployment_variant(
        {
            "baseline_family_control": baseline,
            "baseline_plus_scalar_neighbor": weak,
            "baseline_plus_knn_graph": unsafe,
            "baseline_plus_graph_goal": filler,
            "baseline_plus_graph_history_scalar": filler,
        }
    )
    assert selected["selected_variant"] == "baseline_family_control"


def test_neighbor_rescue_success_requires_nonbaseline_gain_and_easy_safety():
    baseline = {
        "variant": "baseline_family_control",
        "protected": {
            "all_improvement": 0.2,
            "t50_improvement": 0.1,
            "t100_raw_frame_diagnostic_improvement": 0.0,
            "hard_failure_improvement": 0.1,
            "easy_degradation": 0.0,
            "switch_rate": 0.1,
            "harm_over_fallback": -0.1,
        },
    }
    selected = {
        "variant": "baseline_plus_knn_graph",
        "protected": {
            "all_improvement": 0.2,
            "t50_improvement": 0.115,
            "t100_raw_frame_diagnostic_improvement": 0.0,
            "hard_failure_improvement": 0.1,
            "easy_degradation": 0.01,
            "switch_rate": 0.1,
            "harm_over_fallback": -0.1,
        },
    }
    assert s42ck._neighbor_rescue_success(selected, baseline)
    fallback = dict(selected)
    fallback["variant"] = "baseline_family_control"
    assert not s42ck._neighbor_rescue_success(fallback, baseline)


def test_gate_diagnostic_pass_blocks_overclaim():
    metric = {
        "all_improvement": 0.2,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.12,
        "easy_degradation": 0.0,
    }
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "graph_info": {"graph_stats": {"rows_with_neighbors": 100}},
        "variants": {
            "baseline_family_control": {"protected": metric},
            "baseline_plus_scalar_neighbor": {"protected": metric},
            "baseline_plus_knn_graph": {"protected": metric},
            "baseline_plus_graph_goal": {"protected": metric},
            "baseline_plus_graph_history_scalar": {"protected": metric},
        },
        "validation_only_selection": {"selected_variant": "baseline_family_control", "baseline_variant": "baseline_family_control", "test_threshold_tuning": False},
        "neighbor_interaction_rescue_success": False,
        "claim_boundary": {"neighbor_interaction_main_claim_allowed": False, "metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "graph_features_current_and_past_only": True,
            "train_only_feature_normalization": True,
        },
    }
    gate = s42ck._gate(result)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ck_neighbor_interaction_gated_expert_pass_diagnostic_no_overclaim"
