from __future__ import annotations

import numpy as np

from src import stage42_full_waypoint_all_hard_loss_repair as dg


def test_sample_weights_emphasize_hard_long_and_balance_sources() -> None:
    data = {
        "horizon": np.asarray([10, 50, 100, 50]),
        "hard": np.asarray([False, True, False, True]),
        "failure": np.asarray([False, False, True, False]),
        "easy": np.asarray([True, False, False, False]),
        "dataset": np.asarray(["A", "A", "A", "B"]),
    }
    train = np.asarray([True, True, True, True])
    balanced = dg._sample_weights(data, train, "balanced")
    hard = dg._sample_weights(data, train, "all_hard_long_horizon")
    source = dg._sample_weights(data, train, "source_balanced_all_hard_long")
    assert np.allclose(balanced, np.ones(4))
    assert hard[1] > hard[0]
    assert hard[2] > hard[0]
    assert source[3] > hard[3]


def test_weighted_ridge_recovers_linear_target() -> None:
    x = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
    y = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    mask = np.ones(4, dtype=bool)
    weights = np.ones(4, dtype=np.float64)
    coef = dg._fit_weighted_ridge_1d(x, y, mask, weights, lam=1e-6)
    pred = x @ coef
    assert np.max(np.abs(pred - y)) < 1e-3


def test_gate_marks_positive_not_better_than_am() -> None:
    result = {
        "split_stats": {"by_split": {"test": {"rows": 10}}},
        "label_stats": {"test_full_waypoint_rows": 10},
        "model": {
            "candidate_count": 10,
            "selected": {"val_score": 1.0},
            "metrics": {
                "protected_selected_loss_variant": {
                    "rows": 10,
                    "all_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                }
            },
            "bootstrap": {"all": {"bootstrap_n": 1000}},
        },
        "comparison_to_stage42_am": {
            "delta_vs_stage42_am": {
                "all_improvement": -0.01,
                "hard_failure_improvement": -0.01,
            }
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "validation_only_model_selection": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = dg._gate(result)
    assert gate["gates"]["beats_stage42_am_all"] is False
    assert gate["verdict"] == "stage42_dg_full_waypoint_weighted_loss_repair_pass_positive_not_better_than_am"
