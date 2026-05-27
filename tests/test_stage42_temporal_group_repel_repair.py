from __future__ import annotations

import numpy as np

from src import stage42_temporal_group_repel_repair as ez


def test_temporal_weights_tail_moves_late_waypoints_more() -> None:
    weights = ez._temporal_weights("tail", 4, gamma=2.0)
    assert weights.shape == (4,)
    assert weights[0] < weights[-1]
    assert np.isclose(weights[-1], 1.0)


def test_temporal_weights_bell_is_nonzero_and_normalized() -> None:
    weights = ez._temporal_weights("bell", 5)
    assert np.all(weights > 0.0)
    assert np.isclose(np.max(weights), 1.0)


def test_temporal_repel_tail_preserves_early_waypoint_more_than_late() -> None:
    xy = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.01, 0.0], [0.01, 0.0], [0.01, 0.0], [0.01, 0.0]],
        ],
        dtype=np.float32,
    )
    repaired = ez._repel_temporal_rows(
        xy,
        switch=np.asarray([True, False]),
        group_key=np.asarray(["g", "g"], dtype=object),
        normalizer=np.asarray([1.0, 1.0]),
        agent_id=np.asarray([1, 2]),
        current_xy=np.asarray([[0.0, 0.0], [0.01, 0.0]], dtype=np.float32),
        min_sep=0.05,
        strength=1.0,
        weights=ez._temporal_weights("tail", 4, gamma=1.0),
        direction_mode="nearest_current",
    )
    moved = np.linalg.norm(repaired[0] - xy[0], axis=1)
    assert moved[0] < moved[-1]
    assert moved[-1] > 0.0


def test_candidate_grid_has_multiple_temporal_shapes_and_di_uniform_reference() -> None:
    grid = ez._candidate_grid()
    shapes = {row["temporal_kind"] for row in grid}
    assert {"uniform", "tail", "sqrt_tail", "bell", "head"}.issubset(shapes)
    assert any(
        row["temporal_kind"] == "uniform"
        and row["min_sep"] == 0.08
        and row["margin"] == 0.0
        and row["strength"] == 0.5
        for row in grid
    )
    assert len(grid) >= 40


def test_gate_requires_beating_stage42_di_for_promotion() -> None:
    result = {
        "split_stats": {"by_split": {"test": {"rows": 10}}},
        "label_stats": {"test_full_waypoint_rows": 10},
        "repair_family": {"candidate_count": 60, "temporal_shapes": ["uniform", "tail", "sqrt_tail", "bell"]},
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
                "diagnostics": {"final_near_005": 0.01, "base_near_005": 0.02},
                "bootstrap": {"all": {"bootstrap_n": 1000}},
            },
        },
        "comparison_to_stage42_di": {
            "delta_vs_stage42_di": {
                "all_improvement": -0.01,
                "hard_failure_improvement": -0.01,
            },
            "near_delta_vs_stage42_di": -0.001,
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
    gate = ez._gate(result)
    assert gate["gates"]["test_all_positive_vs_floor"] is True
    assert gate["gates"]["beats_stage42_di_all"] is False
    assert gate["verdict"] == "stage42_ez_temporal_group_repel_repair_positive_not_promoted"
