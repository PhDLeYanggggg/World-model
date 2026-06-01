from __future__ import annotations

import numpy as np

from src import stage43_tail_adapter_reviewer_replay as az


def test_parse_allowed_rules() -> None:
    assert az._parse_allowed(["UCY|50", "TrajNet_crowds|10"]) == {("UCY", 50), ("TrajNet_crowds", 10)}


def test_metric_diff_detects_exact_and_nonexact_replay() -> None:
    expected = {
        "full_waypoint_ade_improvement_vs_floor": 0.5,
        "endpoint_fde_improvement_vs_floor": 0.4,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.3,
        "t50_endpoint_fde_improvement_vs_floor": 0.2,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.6,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.7,
    }
    replayed = dict(expected)
    assert az._metric_diff(replayed, expected)["max_abs_diff"] == 0.0
    replayed["t50_full_waypoint_ade_improvement_vs_floor"] = 0.1
    assert abs(az._metric_diff(replayed, expected)["max_abs_diff"] - 0.2) < 1e-9


def test_split_hash_changes_when_rows_change() -> None:
    class Fake:
        feature_names = ["a", "b"]
        x = np.asarray([[1.0, 2.0]], dtype=np.float32)
        horizon = np.asarray([50], dtype=np.int64)
        domain = np.asarray(["UCY"])
        source_file = np.asarray(["u.txt"])
        scene_id = np.asarray(["s"])
        floor_ade = np.asarray([1.0], dtype=np.float32)
        floor_fde = np.asarray([2.0], dtype=np.float32)
        waypoint_delta = np.zeros((1, 4, 2), dtype=np.float32)
        easy = np.asarray([False])
        hard = np.asarray([True])
        failure = np.asarray([False])

    first = az._split_hash(Fake())
    Fake.x = np.asarray([[2.0, 2.0]], dtype=np.float32)
    second = az._split_hash(Fake())
    assert first != second


def test_gate_passes_for_exact_safe_replay() -> None:
    payload = {
        "artifact": str(az.STAGE43_P),
        "artifact_gate": {"verdict": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback"},
        "model_hash_match": True,
        "feature_mean_hash_match": True,
        "feature_std_hash_match": True,
        "split_hashes": {"train": "a" * 64, "val": "b" * 64, "test": "c" * 64},
        "switch_hash": "d" * 64,
        "metric_diff": {"max_abs_diff": 0.0},
        "replay_metrics": {
            "full_waypoint_ade_improvement_vs_floor": 0.5,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.4,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.3,
            "easy_degradation_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "validation_reselection_during_replay": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = az._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["reviewer_replay_passed"] is True


def test_gate_fails_for_replay_metric_drift() -> None:
    payload = {
        "artifact": str(az.STAGE43_P),
        "artifact_gate": {"verdict": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback"},
        "model_hash_match": True,
        "feature_mean_hash_match": True,
        "feature_std_hash_match": True,
        "split_hashes": {"train": "a" * 64, "val": "b" * 64, "test": "c" * 64},
        "switch_hash": "d" * 64,
        "metric_diff": {"max_abs_diff": 0.01},
        "replay_metrics": {
            "full_waypoint_ade_improvement_vs_floor": 0.5,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.4,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.3,
            "easy_degradation_vs_floor": 0.0,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "validation_reselection_during_replay": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = az._gate(payload)
    assert gate["gates"]["replay_metrics_exact"] is False
    assert gate["reviewer_replay_passed"] is False
