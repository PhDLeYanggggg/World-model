from __future__ import annotations

import numpy as np

from src import stage42_context_switchability_family_audit as gk


def test_material_requires_easy_safe_and_one_positive_axis() -> None:
    assert gk._material(
        {
            "all_improvement": 0.011,
            "t50_improvement": -0.1,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
        }
    )
    assert not gk._material(
        {
            "all_improvement": 0.011,
            "t50_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.021,
        }
    )
    assert not gk._material(
        {
            "all_improvement": 0.009,
            "t50_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
        }
    )


def test_gate_passes_for_honest_negative_context_result() -> None:
    payload = {
        "source": gk.SOURCE,
        "summary": {
            "baseline_family_control_val_metric": {"all_improvement": 0.1},
            "families_checked_count": 8,
            "decision": "context_switchability_family_not_supported",
            "material_context_contribution_supported": False,
            "root_cause": "no material lift",
            "next_action": "change source support",
        },
        "family_rows": [
            {"router": {"selection": {"test_threshold_tuning": False}}}
            for _ in range(8)
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "context_main_claim_allowed": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = gk._gate(payload)

    assert gate["verdict"] == "stage42_gk_context_switchability_family_audit_pass"
    assert gate["passed"] == gate["total"]


def test_train_gain_harm_router_uses_val_thresholds_and_selects_positive_gain() -> None:
    rng = np.random.default_rng(7)
    n = 90
    raw = np.c_[rng.normal(size=n), np.ones(n)].astype(np.float32)
    split = np.array(["train"] * 40 + ["val"] * 25 + ["test"] * 25)
    base = np.ones(n, dtype=np.float32)
    candidate = base.copy()
    helpful = raw[:, 0] > 0.2
    candidate[helpful] = 0.5
    candidate[~helpful] = 1.4
    data = {
        "horizon": np.full(n, 50),
        "hard": helpful,
        "failure": helpful,
        "easy": ~helpful,
    }

    result = gk._train_gain_harm_router(
        raw_features=raw,
        base_ade=base,
        candidate_ade=candidate,
        split=split,
        data=data,
    )

    assert result["selection"]["test_threshold_tuning"] is False
    assert result["selection"]["candidates_checked"] > 0
    assert result["test_metric_vs_baseline_family"]["easy_degradation"] <= 0.02
