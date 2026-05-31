from __future__ import annotations

import numpy as np

from src import stage43_t100_source_stability_guard as r


def test_source_stability_blocks_single_source_family() -> None:
    class Fake:
        source_file = np.asarray(["/tmp/UCY/one.txt"] * 8)
        horizon = np.asarray([100] * 8)
        floor_ade = np.ones(8, dtype=np.float32)
        easy = np.zeros(8, dtype=bool)

    candidate = np.full(8, 0.8, dtype=np.float32)
    table, allowed = r._h100_source_stability_table(
        Fake(),
        candidate,
        min_family_rows=4,
        min_source_rows=4,
        min_source_count=2,
        min_improvement=0.0,
        max_easy_degradation=0.02,
    )
    assert ("UCY", 100) not in allowed
    assert table["UCY|100"]["reason"] == "blocked_insufficient_validation_source_count"
    assert table["UCY|100"]["supported_source_count"] == 1


def test_source_stability_allows_two_safe_sources() -> None:
    class Fake:
        source_file = np.asarray(["/tmp/UCY/one.txt"] * 4 + ["/tmp/UCY/two.txt"] * 4)
        horizon = np.asarray([100] * 8)
        floor_ade = np.ones(8, dtype=np.float32)
        easy = np.zeros(8, dtype=bool)

    candidate = np.asarray([0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7], dtype=np.float32)
    table, allowed = r._h100_source_stability_table(
        Fake(),
        candidate,
        min_family_rows=4,
        min_source_rows=4,
        min_source_count=2,
        min_improvement=0.0,
        max_easy_degradation=0.02,
    )
    assert ("UCY", 100) in allowed
    assert table["UCY|100"]["reason"] == "allowed_by_source_stable_validation"
    assert table["UCY|100"]["safe_supported_source_count"] == 2


def test_gate_passes_when_q_false_positive_is_blocked() -> None:
    payload = {
        "source": r.SOURCE,
        "stage43_p_precondition": {
            "verdict": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
            "replay_exact": True,
        },
        "stage43_q_reference": {
            "verdict": "stage43_q_t100_guarded_trial_honest_blocker",
        },
        "result_source": "fresh_validation_source_stable_t100_guard",
        "training_protocol": {"test_threshold_tuning": False},
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
        },
        "selected_source_stable_trial": {
            "validation_h100_source_stability_table": {"UCY|100": {}},
        },
        "t100_source_stability_guard": {
            "source_stability_blocks_stage43_q_false_positive": True,
            "deploy_h100_adapter": False,
        },
        "overall_full_test_metrics": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
            "easy_degradation_vs_floor": 0.0,
        },
        "delta_vs_stage43_p": {
            "t50_delta": 0.0,
            "hard_failure_delta": 0.0,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = r._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_h100_adapter"] is False
