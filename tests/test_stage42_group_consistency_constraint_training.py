from __future__ import annotations

import numpy as np

from src.stage42_group_consistency_constraint_training import _constraint_weights, _deployment_decision, _gate


def _metric(all_imp: float, t50: float, hard: float, easy: float) -> dict:
    return {
        "rows": 100,
        "all_improvement": all_imp,
        "t10_improvement": all_imp,
        "t25_improvement": all_imp,
        "t50_improvement": t50,
        "t100_raw_frame_diagnostic_improvement": 0.1,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
        "switch_rate": 0.2,
        "harm_over_fallback": -0.1,
    }


def test_stage42_eu_constraint_weights_emphasize_group_risk_and_normalize_train_mean() -> None:
    data = {
        "horizon": np.asarray([10, 50, 100, 50]),
        "hard": np.asarray([False, True, False, True]),
        "failure": np.asarray([False, False, True, False]),
        "easy": np.asarray([True, False, False, False]),
    }
    group_risk = {
        "risk": np.asarray([0.0, 2.0, 5.0, 1.0]),
    }
    graph = np.zeros((4, 40), dtype=np.float32)
    train_mask = np.asarray([True, True, True, False])

    weights = _constraint_weights(data, group_risk, graph, train_mask, "group_unsafe_hard_weighted")

    assert np.isclose(float(np.mean(weights[train_mask])), 1.0)
    assert weights[2] > weights[0]
    assert weights[1] > weights[0]


def test_stage42_eu_deployment_decision_requires_beating_stage42_di() -> None:
    metric = _metric(0.25, 0.22, 0.24, -0.1)
    promotes = _deployment_decision(
        metric,
        {
            "delta_vs_stage42_am": {"all_improvement": 0.01},
            "delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.01},
        },
    )
    diagnostic = _deployment_decision(
        metric,
        {
            "delta_vs_stage42_am": {"all_improvement": 0.01},
            "delta_vs_stage42_di": {"all_improvement": -0.01, "hard_failure_improvement": 0.01},
        },
    )

    assert promotes["promote_group_constraint_training"] is True
    assert promotes["decision"] == "promote_stage42_eu_group_constraint_training"
    assert diagnostic["promote_group_constraint_training"] is False
    assert diagnostic["diagnostic_positive"] is True


def test_stage42_eu_gate_passes_for_promotable_payload() -> None:
    payload = {
        "split_stats": {"by_split": {"test": {"rows": 100}}},
        "label_stats": {"test_full_waypoint_rows": 100},
        "group_risk_stats": {"train_close008_rate": 0.1},
        "training": {
            "candidate_count": 20,
            "selected": {"val_score": 1.0},
        },
        "group_repaired_selected": {
            "selected": {"val_score": 1.0},
            "test": {
                "metric_vs_floor": _metric(0.25, 0.22, 0.24, -0.1),
                "diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01},
            },
        },
        "comparison_to_prior": {
            "delta_vs_stage42_am": {"all_improvement": 0.01},
            "delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.01},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_model_and_repair_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": True,
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
    assert gate["verdict"] == "stage42_eu_group_consistency_constraint_training_pass_promotable"
