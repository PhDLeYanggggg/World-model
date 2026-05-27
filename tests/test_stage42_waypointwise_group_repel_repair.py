from __future__ import annotations

import numpy as np

from src import stage42_waypointwise_group_repel_repair as fa


def test_smooth_offsets_preserves_shape_and_spreads_middle() -> None:
    offsets = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    smoothed = fa._smooth_offsets(offsets)
    assert smoothed.shape == offsets.shape
    assert smoothed[1, 0] == 1.0
    assert smoothed[0, 0] == 0.0


def test_waypointwise_repel_only_moves_close_waypoint_when_unsmoothed() -> None:
    xy = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.50, 0.0]],
            [[1.0, 0.0], [0.01, 0.0], [1.50, 0.0]],
        ],
        dtype=np.float32,
    )
    repaired = fa._repel_waypointwise_rows(
        xy,
        switch=np.asarray([True, False]),
        group_key=np.asarray(["g", "g"], dtype=object),
        normalizer=np.asarray([1.0, 1.0]),
        agent_id=np.asarray([1, 2]),
        current_xy=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        min_sep=0.05,
        strength=1.0,
        weights=np.ones(3, dtype=np.float32),
        smooth=False,
        cap_scale=1.0,
    )
    moved = np.linalg.norm(repaired[0] - xy[0], axis=1)
    assert moved[0] == 0.0
    assert moved[1] > 0.0
    assert moved[2] == 0.0


def test_candidate_grid_contains_smoothed_and_unsmoothed_variants() -> None:
    grid = fa._candidate_grid()
    assert len(grid) >= 40
    assert {row["smooth"] for row in grid} == {False, True}
    assert {"uniform", "tail", "sqrt_tail", "bell"}.issubset({row["temporal_kind"] for row in grid})


def test_gate_keeps_positive_not_promoted_without_near_safety() -> None:
    result = {
        "split_stats": {"by_split": {"test": {"rows": 10}}},
        "label_stats": {"test_full_waypoint_rows": 10},
        "repair_family": {"candidate_count": 72},
        "repair": {
            "selected": {"val_score": 1.0},
            "test": {
                "metric_vs_floor": {
                    "rows": 10,
                    "all_improvement": 0.1,
                    "t50_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                },
                "diagnostics": {"final_near_005": 0.03, "base_near_005": 0.02},
                "bootstrap": {"all": {"bootstrap_n": 1000}},
            },
        },
        "comparison_to_prior": {
            "delta_vs_stage42_di": {
                "all_improvement": 0.01,
                "hard_failure_improvement": 0.01,
            },
            "near_delta_vs_stage42_di": 0.01,
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
    gate = fa._gate(result)
    assert gate["gates"]["beats_stage42_di_all"] is True
    assert gate["gates"]["near_not_worse_than_stage42_di"] is False
    assert gate["verdict"] == "stage42_fa_waypointwise_group_repel_repair_positive_not_promoted"
