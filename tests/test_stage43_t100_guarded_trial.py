from __future__ import annotations

import numpy as np

from src import stage43_t100_guarded_trial as q


def test_select_h100_rules_allows_only_safe_supported_family() -> None:
    class Fake:
        source_file = np.asarray(["/tmp/UCY/a.txt"] * 4 + ["/tmp/TrajNet/Train/crowds/z.txt"] * 4)
        horizon = np.asarray([100] * 8)
        floor_ade = np.ones(8, dtype=np.float32)
        easy = np.asarray([False, False, False, False, False, False, True, True])

    candidate = np.asarray([0.7, 0.7, 0.7, 0.7, 0.1, 0.1, 1.1, 1.1], dtype=np.float32)
    table, allowed = q._select_h100_rules(
        Fake(),
        candidate,
        min_support_rows=4,
        min_improvement=0.0,
        max_easy_degradation=0.02,
    )
    assert ("UCY", 100) in allowed
    assert ("TrajNet_crowds", 100) not in allowed
    assert table["TrajNet_crowds|100"]["reason"] == "blocked_validation_easy_harm"


def test_combine_with_h100_preserves_base_non_h100_rows() -> None:
    class Fake:
        source_file = np.asarray(["/tmp/UCY/a.txt", "/tmp/UCY/a.txt", "/tmp/ETH/seq.txt"])
        horizon = np.asarray([50, 100, 100])

    base_ade = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    base_fde = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
    base_switch = np.asarray([True, False, False])
    candidate_ade = np.asarray([9.0, 0.5, 0.4], dtype=np.float32)
    candidate_fde = np.asarray([9.0, 1.0, 0.8], dtype=np.float32)
    selected_ade, selected_fde, switch = q._combine_with_h100(
        Fake(),
        base_ade,
        base_fde,
        base_switch,
        candidate_ade,
        candidate_fde,
        {("UCY", 100)},
    )
    assert selected_ade.tolist() == [1.0, 0.5, 1.0]
    assert selected_fde.tolist() == [2.0, 1.0, 2.0]
    assert switch.tolist() == [True, True, False]


def test_gate_passes_honest_blocker_without_deploying_t100() -> None:
    payload = {
        "source": q.SOURCE,
        "stage43_p_precondition": {
            "verdict": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
            "replay_exact": True,
        },
        "result_source": "fresh_validation_selected_t100_guarded_trial",
        "training_protocol": {"selection_data": "validation_only", "test_threshold_tuning": False},
        "no_leakage": {
            "test_threshold_tuning": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
        },
        "selected_t100_trial": {"validation_h100_support_table": {"UCY|100": {}}, "h100_allowed_rules": []},
        "overall_full_test_metrics": {
            "full_waypoint_ade_improvement_vs_floor": 0.5,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.5,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.47,
            "easy_degradation_vs_floor": 0.0,
        },
        "delta_vs_stage43_p": {
            "t50_delta": 0.0,
            "hard_failure_delta": 0.0,
        },
        "by_horizon": {"100": {"easy_degradation_vs_floor": 0.0}},
        "t100_guarded_trial": {
            "t100_positive_success": False,
            "deploy_h100_adapter": False,
            "blocker": "validation_selected_no_h100_family_with_enough_safe_margin",
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = q._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["deploy_h100_adapter"] is False
    assert gate["verdict"] == "stage43_q_t100_guarded_trial_honest_blocker"
