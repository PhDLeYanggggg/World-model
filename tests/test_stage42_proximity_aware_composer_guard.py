from __future__ import annotations

import numpy as np

from src import stage42_proximity_aware_composer_guard as cq


def _toy_endpoint_full() -> tuple[dict, dict, np.ndarray]:
    labels = {
        "normalizer": np.ones(2, dtype=np.float64),
        "horizon": np.asarray([50, 50]),
        "hard": np.asarray([True, True]),
        "failure": np.asarray([False, False]),
        "easy": np.asarray([False, False]),
        "domain": np.asarray(["toy", "toy"], dtype=object),
        "waypoint_valid": np.ones((2, 1), dtype=bool),
        "waypoint_xy": np.asarray([[[0.0, 0.0]], [[0.20, 0.0]]], dtype=np.float64),
    }
    endpoint_xy = np.asarray([[[0.0, 0.0]], [[0.20, 0.0]]], dtype=np.float64)
    full_xy = np.asarray([[[0.0, 0.0]], [[0.01, 0.0]]], dtype=np.float64)
    endpoint = {
        "labels": labels,
        "selected_xy": endpoint_xy,
        "floor_xy": endpoint_xy,
        "selected_ade": np.asarray([1.0, 1.0], dtype=np.float64),
        "floor_ade": np.asarray([1.1, 1.1], dtype=np.float64),
        "selected_fde": np.asarray([1.0, 1.0], dtype=np.float64),
        "floor_fde": np.asarray([1.1, 1.1], dtype=np.float64),
        "switch": np.asarray([False, False]),
    }
    full = {
        "selected_xy": full_xy,
        "selected_ade": np.asarray([0.8, 0.8], dtype=np.float64),
        "floor_ade": np.asarray([1.1, 1.1], dtype=np.float64),
        "selected_fde": np.asarray([0.8, 0.8], dtype=np.float64),
        "floor_fde": np.asarray([1.1, 1.1], dtype=np.float64),
    }
    keys = np.asarray(["scene|1|50", "scene|1|50"], dtype=object)
    return endpoint, full, keys


def test_proximity_guard_turns_off_harmful_close_row() -> None:
    endpoint, full, keys = _toy_endpoint_full()
    ev = cq._apply_proximity_guard(endpoint, full, keys, {"toy|50": True}, min_sep=0.05, margin=0.0)
    assert ev["guarded_off"] == 2
    assert not ev["use_full"].any()


def test_score_penalizes_near_collision_delta() -> None:
    metric = {
        "all_improvement": 0.02,
        "t50_improvement": 0.02,
        "t100_raw_frame_diagnostic_improvement": 0.0,
        "hard_failure_improvement": 0.02,
        "easy_degradation": 0.0,
    }
    assert cq._score(metric, near_collision_delta=-0.001) > cq._score(metric, near_collision_delta=0.01)


def test_gate_passes_safe_positive_policy() -> None:
    payload = {
        "inputs": {
            "stage42_co": {"stage42_co_gate": {"passed": 14, "total": 14}},
            "stage42_cp": {"stage42_cp_gate": {"passed": 14, "total": 14}},
        },
        "policy_selection": {"selected_on": "validation_only", "test_evaluated_once": True},
        "test_eval": {
            "metric_vs_endpoint_ade": {
                "all_improvement": 0.01,
                "t50_improvement": 0.01,
                "t100_raw_frame_diagnostic_improvement": 0.02,
                "hard_failure_improvement": 0.01,
                "easy_degradation": 0.001,
            }
        },
        "test_joint_safety": {
            "composer_minus_endpoint": {"near_collision_rate_005_delta": -0.001},
            "composer_minus_floor": {"near_collision_rate_005_delta": -0.001},
        },
        "bootstrap_vs_endpoint_ade": {
            "all": {"low": 0.001},
            "t50": {"low": 0.001},
        },
        "paper_file_status": [{"contains_stage42_cq": True}],
        "no_leakage": {"test_threshold_tuning": False, "guard_uses_future_labels": False},
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = cq._gate(payload)
    assert gate["verdict"] == "stage42_cq_proximity_aware_composer_guard_pass"
    assert gate["passed"] == gate["total"]
