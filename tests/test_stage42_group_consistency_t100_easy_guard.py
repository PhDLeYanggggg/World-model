from __future__ import annotations

import numpy as np

from src import stage42_group_consistency_t100_easy_guard as hr


def test_keep_t100_slice_requires_positive_gain_and_nonharm_easy() -> None:
    assert hr._keep_t100_slice(0.1, -0.01) is True
    assert hr._keep_t100_slice(0.1, 0.01) is False
    assert hr._keep_t100_slice(0.0, -0.01) is False


def test_easy_degradation_is_positive_when_selected_hurts_easy() -> None:
    selected = np.asarray([3.0, 5.0, 1.0])
    floor = np.asarray([2.0, 4.0, 1.0])
    mask = np.asarray([True, True, False])
    assert hr._easy_degradation(selected, floor, mask) > 0.0


def test_gate_passes_for_validation_only_t100_guard_payload() -> None:
    metric = {
        "rows": 10,
        "all_improvement": 0.2,
        "t50_improvement": 0.1,
        "t100_raw_frame_diagnostic_improvement": 0.03,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.1,
    }
    payload = {
        "hq_input": {"exists": True},
        "pre_guard": {"t100_easy_degradation": 0.03},
        "guarded": {
            "source": "fresh_validation_only_domain_t100_easy_guard",
            "guarded_slices": {"TrajNet|100": {}},
            "kept_slices": {"UCY|100": {}},
            "t100_easy_degradation": 0.0,
            "metric": metric,
            "by_domain": {"UCY": {"all_improvement": 0.2, "t50_improvement": 0.1}},
            "diagnostics": {"pre_guard_near_005": 0.02, "post_guard_near_005": 0.01},
        },
        "uses_test_metrics_for_guard": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "source_overlap_pass": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = hr._gate(payload)
    assert gate["verdict"] == "stage42_hr_t100_easy_guard_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_t100_easy_remains_over_two_percent() -> None:
    metric = {
        "rows": 10,
        "all_improvement": 0.2,
        "t50_improvement": 0.1,
        "t100_raw_frame_diagnostic_improvement": 0.03,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.1,
    }
    payload = {
        "hq_input": {"exists": True},
        "pre_guard": {"t100_easy_degradation": 0.03},
        "guarded": {
            "source": "fresh_validation_only_domain_t100_easy_guard",
            "guarded_slices": {"TrajNet|100": {}},
            "kept_slices": {},
            "t100_easy_degradation": 0.025,
            "metric": metric,
            "by_domain": {"UCY": {"all_improvement": 0.2, "t50_improvement": 0.1}},
            "diagnostics": {"pre_guard_near_005": 0.02, "post_guard_near_005": 0.01},
        },
        "uses_test_metrics_for_guard": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "source_overlap_pass": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = hr._gate(payload)
    assert gate["gates"]["t100_easy_repaired_under_2pct"] is False
