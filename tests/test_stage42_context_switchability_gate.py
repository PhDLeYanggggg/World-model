from __future__ import annotations

import numpy as np

from src import stage42_context_switchability_gate as dc


def _data(n: int) -> dict[str, np.ndarray]:
    return {
        "current_x": np.linspace(0.0, 1.0, n).astype(np.float32),
        "current_y": np.linspace(1.0, 2.0, n).astype(np.float32),
        "scale": np.ones(n, dtype=np.float32),
        "history_scalar": np.ones((n, 3), dtype=np.float32),
        "horizon": np.array([50, 50, 10, 25, 50, 100, 50, 25, 10, 50, 100, 50], dtype=np.int64)[:n],
        "hard": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)[:n],
        "failure": np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=bool)[:n],
        "easy": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)[:n],
    }


def test_switch_features_are_past_and_prediction_shaped() -> None:
    n = 5
    data = _data(n)
    baseline_xy = np.zeros((n, 4, 2), dtype=np.float32)
    candidate_xy = np.ones((n, 4, 2), dtype=np.float32)
    raw_context = np.ones((n, 120), dtype=np.float32)
    features = dc._switch_features(data, baseline_xy, candidate_xy, raw_context)
    assert features.shape == (n, 2 + 1 + 2 + 2 + 3 + 96)
    assert np.isfinite(features).all()


def test_choose_switch_policy_uses_validation_and_keeps_easy_guard() -> None:
    n = 12
    data = _data(n)
    split = np.array(["train"] * 4 + ["val"] * 4 + ["test"] * 4)
    floor = np.ones(n, dtype=np.float64)
    baseline = np.ones(n, dtype=np.float64) * 0.8
    candidate = baseline.copy()
    candidate[[4, 6, 8, 10]] = 0.5
    predicted_gain = np.linspace(0.0, 1.0, n)
    policy = dc._choose_switch_policy(
        "candidate",
        predicted_gain,
        baseline,
        candidate,
        floor,
        data,
        split,
    )
    assert policy["test_threshold_tuning"] is False
    assert policy["test_metric"]["easy_degradation"] <= dc.EASY_LIMIT
    assert "validation_candidates" in policy


def test_gate_passes_for_honest_negative_context_switchability() -> None:
    payload = {
        "split_stats": {
            "by_split": {"test": {"rows": 47458}},
            "source_overlap_pass": True,
        },
        "baseline_family_control": {
            "protected_metric": {"all_improvement": 0.2, "t50_improvement": 0.1}
        },
        "candidate_results": {f"candidate_{i}": {} for i in range(5)},
        "selected_context_switchability_policy": {
            "decision": "context_switchability_not_supported",
            "context_switchability_supported": False,
            "bootstrap": {
                "all": {"bootstrap_n": 1000},
                "t50": {"bootstrap_n": 1000},
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "gain_label_train_only_for_model_fit": True,
            "validation_only_threshold_selection": True,
            "test_threshold_tuning": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "source_overlap_pass": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = dc._gate(payload)
    assert gate["verdict"] == "stage42_dc_context_switchability_gate_pass"
    assert gate["passed"] == gate["total"]
