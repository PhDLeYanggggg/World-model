from __future__ import annotations

import numpy as np

from src import stage43_tail_horizon_waypoint_adapter as p


def test_ridge_fit_recovers_simple_linear_map() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.asarray([[1.0], [3.0], [5.0], [7.0]], dtype=np.float32)
    w = p._ridge_fit(x, y, 1e-6)
    pred = p._ridge_predict(np.asarray([[4.0]], dtype=np.float32), w)
    assert abs(float(pred[0, 0]) - 9.0) < 1e-3


def test_h100_contract_blocks_supported_unsafe_h100() -> None:
    class Fake:
        source_file = np.asarray(["/tmp/UCY/a.txt"] * 4 + ["/tmp/TrajNet/Train/crowds/z.txt"] * 4)
        horizon = np.asarray([100, 100, 100, 100, 100, 100, 100, 100])
        floor_ade = np.ones(8, dtype=np.float32)
        easy = np.asarray([False, False, False, False, True, True, True, True])

    candidate = np.asarray([0.8, 0.8, 0.8, 0.8, 1.3, 1.3, 1.3, 1.3], dtype=np.float32)
    table, allowed, contract = p._select_support_rules(
        Fake(),
        candidate,
        min_support_rows=4,
        min_improvement=0.0,
        max_easy_degradation=0.02,
        require_all_supported_h100_safe=True,
    )
    assert ("UCY", 100) not in allowed
    assert contract["allow_h100"] is False
    assert table["UCY|100"]["reason"] == "blocked_h100_global_validation_contract"


def test_gate_requires_positive_stage43_o_delta_and_t100_no_overclaim() -> None:
    payload = {
        "source": p.SOURCE,
        "stage43_o_precondition": {"verdict": "stage43_o_safe_repair_pass_t100_fallback_not_positive"},
        "result_source": "fresh_train_val_selected_tail_horizon_adapter",
        "training_protocol": {"selection_data": "validation_only", "test_threshold_tuning": False},
        "no_leakage": {"test_threshold_tuning": False, "future_waypoint_input": False, "future_waypoint_label_eval_only": True},
        "overall_full_test_metrics": {
            "full_waypoint_ade_improvement_vs_floor": 0.3,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.2,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.25,
            "easy_degradation_vs_floor": 0.0,
        },
        "by_horizon": {"100": {"easy_degradation_vs_floor": 0.0}},
        "delta_vs_stage43_o": {
            "full_waypoint_ade_improvement_delta": 0.1,
            "t50_delta": 0.1,
            "hard_failure_delta": 0.1,
        },
        "by_source_summary": {"negative_source_count": 0},
        "bootstrap_ci": {
            "metrics": {
                "t50_full_waypoint_ade_improvement_vs_floor": {"low": 0.05},
            }
        },
        "claim_boundary": {
            "t100_positive_success": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = p._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_tail_horizon_adapter"] is True
