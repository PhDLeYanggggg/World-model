from __future__ import annotations

import numpy as np

from src import stage42_group_consistency_full_waypoint_repair as di


def test_min_group_distance_deduplicates_agents_and_normalizes() -> None:
    xy = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[3.0, 4.0], [3.0, 4.0]],
            [[100.0, 100.0], [100.0, 100.0]],
            [[101.0, 100.0], [101.0, 100.0]],
        ],
        dtype=np.float32,
    )
    key = np.asarray(["a", "a", "b", "b"], dtype=object)
    scale = np.asarray([5.0, 5.0, 2.0, 2.0], dtype=np.float64)
    agent = np.asarray([1, 2, 1, 2], dtype=np.int64)
    out = di._min_group_distance_fast(xy, key, scale, agent)
    assert np.allclose(out[:2], [1.0, 1.0])
    assert np.allclose(out[2:], [0.5, 0.5])


def test_repair_subset_falls_back_unsafe_rows() -> None:
    data = {
        "scale": np.asarray([1.0, 1.0]),
        "agent_id": np.asarray([1, 2]),
        "current_x": np.asarray([0.0, 1.0]),
        "current_y": np.asarray([0.0, 0.0]),
        "horizon": np.asarray([50, 50]),
        "hard": np.asarray([False, False]),
        "failure": np.asarray([False, False]),
        "easy": np.asarray([False, False]),
    }
    labels = {
        "waypoint_xy": np.asarray(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "waypoint_valid": np.ones((2, 2), dtype=bool),
    }
    floor = labels["waypoint_xy"].copy()
    pred = np.asarray(
        [
            [[0.5, 0.0], [0.5, 0.0]],
            [[0.51, 0.0], [0.51, 0.0]],
        ],
        dtype=np.float32,
    )
    base = pred.copy()
    switch = np.asarray([True, True])
    key = np.asarray(["g", "g"], dtype=object)
    out = di._repair_subset(
        np.asarray([0, 1]),
        {"mode": "fallback_unsafe", "min_sep": 0.05, "margin": 0.0},
        data,
        labels,
        floor,
        pred,
        base,
        switch,
        key,
    )
    assert out["diagnostics"]["unsafe_rows"] == 2
    assert out["switch"].tolist() == [False, False]
    assert out["diagnostics"]["final_near_005"] == 0.0


def test_gate_marks_positive_not_promotable_without_stage42_am_gain() -> None:
    result = {
        "split_stats": {"by_split": {"test": {"rows": 10}}},
        "label_stats": {"test_full_waypoint_rows": 10},
        "group_schema": {"agent_deduplication": True},
        "repair": {
            "candidate_count": 42,
            "selected": {"val_score": 1.0},
            "test": {
                "metric_vs_floor": {
                    "rows": 10,
                    "all_improvement": 0.1,
                    "t50_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                },
                "diagnostics": {"final_near_005": 0.01, "base_near_005": 0.02},
                "bootstrap": {"all": {"bootstrap_n": 1000}},
            },
        },
        "comparison_to_prior": {
            "delta_vs_stage42_am": {
                "all_improvement": -0.01,
                "hard_failure_improvement": -0.01,
            }
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "group_features_predicted_rollout_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = di._gate(result)
    assert gate["gates"]["test_all_positive_vs_floor"] is True
    assert gate["gates"]["beats_stage42_am_all"] is False
    assert gate["verdict"] == "stage42_di_group_consistency_full_waypoint_repair_pass_positive_not_promotable"
