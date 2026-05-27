from __future__ import annotations

import numpy as np

from src.stage42_context_gain_router import _fit_ridge_vector, _gate, _score, _train_gain_router


def test_stage42_el_score_rewards_gain_and_penalizes_easy_harm() -> None:
    good = {
        "all_improvement": 0.02,
        "t50_improvement": 0.03,
        "hard_failure_improvement": 0.02,
        "easy_degradation": 0.0,
        "switch_rate": 0.2,
    }
    bad = dict(good)
    bad["easy_degradation"] = 0.05

    assert _score(good) > _score(bad)


def test_stage42_el_gain_router_can_learn_safe_switch_on_synthetic_data() -> None:
    rng = np.random.default_rng(7)
    n = 240
    x = rng.normal(size=(n, 3)).astype(np.float32)
    split = np.array(["train"] * 120 + ["val"] * 60 + ["test"] * 60)
    base = np.ones(n, dtype=np.float64)
    candidate = base.copy()
    helpful = x[:, 0] > 0.2
    candidate[helpful] -= 0.25
    candidate[~helpful] += 0.05
    data = {
        "horizon": np.array([50] * n),
        "hard": helpful,
        "failure": helpful,
        "easy": ~helpful,
    }

    result = _train_gain_router(
        name="synthetic_context",
        raw_router_features=x,
        base_ade=base,
        candidate_ade=candidate,
        split=split,
        data=data,
    )

    assert result["validation_selection"]["source"] == "validation_only"
    assert result["validation_selection"]["test_threshold_tuning"] is False
    assert result["test_metric_vs_baseline_family"]["all_improvement"] > 0
    assert result["test_metric_vs_baseline_family"]["easy_degradation"] <= 0.02


def test_stage42_el_gate_passes_with_bounded_positive_or_negative_claim() -> None:
    payload = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "baseline_family_control": {"policy_slice_count": 1},
        "routers": {
            f"c{i}": {"validation_selection": {"source": "validation_only", "test_threshold_tuning": False}}
            for i in range(3)
        },
        "summary": {"context_increment_verdict": "stage42_el_context_gain_router_not_supported"},
        "positive_context_gain_routers": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_selected_thresholds": True,
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
