from __future__ import annotations

import numpy as np

from src.stage42_objective_level_proximity_training import _deployment_decision, _gate, _objective_weights


def _metric(all_imp: float, hard: float, easy: float) -> dict:
    return {
        "rows": 20,
        "all_improvement": all_imp,
        "t10_improvement": all_imp,
        "t25_improvement": all_imp,
        "t50_improvement": all_imp,
        "t100_raw_frame_diagnostic_improvement": all_imp,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
        "switch_rate": 0.2,
        "harm_over_fallback": -0.1,
    }


def test_objective_weights_use_future_labels_as_loss_weight_signal_only() -> None:
    data = {
        "horizon": np.asarray([10, 50, 100, 50]),
        "hard": np.asarray([False, True, False, True]),
        "failure": np.asarray([False, False, True, False]),
        "easy": np.asarray([True, False, False, False]),
    }
    signals = {
        "risk": np.asarray([0.0, 1.0, 3.0, 0.5]),
    }
    graph = np.zeros((4, 40), dtype=np.float32)
    graph[:, 1] = [2.0, 0.5, 0.3, 1.5]
    train = np.asarray([True, True, True, False])

    weights = _objective_weights(data, signals, graph, train, "label_proximity_hard_long_objective")

    assert np.isclose(float(np.mean(weights[train])), 1.0)
    assert weights[2] > weights[0]
    assert weights[1] > weights[0]


def test_deployment_decision_requires_beating_di_and_fb_and_near_safety() -> None:
    metric = _metric(0.25, 0.24, 0.0)
    promotes = _deployment_decision(
        metric,
        {
            "delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.01},
            "delta_vs_stage42_fb": {"all_improvement": 0.01},
            "near_delta_vs_stage42_di": -0.001,
        },
        {"protected_near": {"near_005": 0.01}},
    )
    diagnostic = _deployment_decision(
        metric,
        {
            "delta_vs_stage42_di": {"all_improvement": -0.01, "hard_failure_improvement": 0.01},
            "delta_vs_stage42_fb": {"all_improvement": 0.01},
            "near_delta_vs_stage42_di": -0.001,
        },
        {"protected_near": {"near_005": 0.01}},
    )

    assert promotes["promote_objective_level_training"] is True
    assert promotes["decision"] == "promote_stage42_fc_objective_level_training"
    assert diagnostic["promote_objective_level_training"] is False
    assert diagnostic["diagnostic_positive"] is True


def test_gate_accepts_promotable_objective_payload() -> None:
    payload = {
        "source": "fresh_stage42_objective_level_proximity_training",
        "split_stats": {"by_split": {"test": {"rows": 100}}},
        "label_stats": {"test_full_waypoint_rows": 100},
        "objective_signal_stats": {"train_future_close008_rate": 0.1},
        "model": {
            "candidate_count": 25,
            "selected": {"val_score": 1.0},
            "metrics": {"protected_selected_candidate": _metric(0.25, 0.24, 0.0)},
        },
        "comparison_to_prior": {
            "delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.01},
            "delta_vs_stage42_fb": {"all_improvement": 0.01},
            "near_delta_vs_stage42_di": -0.001,
        },
        "no_leakage": {
            "future_waypoint_labels_loss_only": True,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "validation_only_model_selection": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_fc_objective_level_proximity_training_pass_promotable"
