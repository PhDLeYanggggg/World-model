import numpy as np

from src import stage42_full_waypoint_dynamics as s42c


def test_stage42c_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42c.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42c_safe_improvement_positive_when_selected_lower() -> None:
    selected = np.asarray([0.5, 0.75, 1.0])
    floor = np.asarray([1.0, 1.0, 1.0])
    mask = np.asarray([True, True, True])
    assert s42c._safe_improvement(selected, floor, mask) > 0.0


def test_stage42c_metrics_easy_degradation_sign() -> None:
    selected = np.asarray([0.5, 1.5, 0.5, 1.5])
    floor = np.asarray([1.0, 1.0, 1.0, 1.0])
    labels = {
        "horizon": np.asarray([50, 50, 100, 100]),
        "hard": np.asarray([True, False, True, False]),
        "failure": np.asarray([False, False, False, False]),
        "easy": np.asarray([False, True, False, True]),
        "domain": np.asarray(["A", "A", "B", "B"]),
    }
    switch = np.asarray([True, True, True, True])
    metrics = s42c._metrics(selected, floor, labels, switch)
    assert metrics["hard_failure_improvement"] > 0.0
    assert metrics["easy_degradation"] > 0.0


def test_stage42c_gate_requires_full_waypoint_model() -> None:
    metric = {
        "rows": 10,
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "by_domain": {
            "ETH_UCY": {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1},
            "TrajNet": {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1},
        },
    }
    result = {
        "label_reconstruction": {"splits": {"test": {"rows": 10}}},
        "full_waypoint_training_result": {"best_name": "full_trajectory_ensemble"},
        "comparisons": {
            "endpoint_only_final_fde": {},
            "m3w_neural_v1_composite_tail_linear_bridge": {},
            "learned_waypoint_shape_bridge": {},
            "graph_interaction_group_consistency": {},
            "ungated_full_waypoint_transformer": {},
            "full_waypoint_transformer_protected": {
                "ade": metric,
                "fde": metric,
                "joint": {"near_collision_delta_005": 0.0},
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42c._gate(result)
    assert gate["passed"] == gate["total"]
