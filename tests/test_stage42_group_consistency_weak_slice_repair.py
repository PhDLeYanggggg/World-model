from __future__ import annotations

import numpy as np

from src import stage42_group_consistency_weak_slice_repair as hq


def test_hp_ucy_before_extracts_domain_ucy() -> None:
    payload = {
        "breakdown": {
            "by_domain": [
                {"name": "domain:TrajNet", "ade_all_improvement": 0.3},
                {"name": "domain:UCY", "ade_all_improvement": 0.0, "ade_t50_improvement": 0.0},
            ]
        }
    }
    row = hq._hp_ucy_before(payload)
    assert row["name"] == "domain:UCY"
    assert row["ade_all_improvement"] == 0.0


def test_easy_degradation_on_mask_is_negative_when_selected_improves_easy() -> None:
    selected = np.asarray([1.0, 2.0, 5.0])
    floor = np.asarray([2.0, 4.0, 5.0])
    mask = np.asarray([True, True, False])
    assert hq._easy_degradation_on_mask(selected, floor, mask) < 0.0


def test_gate_requires_ucy_repair_without_metric_overclaim() -> None:
    metric = {
        "rows": 10,
        "all_improvement": 0.2,
        "t50_improvement": 0.1,
        "t100_raw_frame_diagnostic_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.1,
    }
    payload = {
        "hp_input": {"exists": True},
        "hp_ucy_before": {"ade_all_improvement": 0.0},
        "ucy_supported_repair": {
            "source": "fresh_ucy_internal_validation_supported_repair",
            "test": {
                "metric": metric,
                "by_domain": {"UCY": metric},
                "diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01},
                "t100_easy_rows": 5,
            },
            "no_leakage": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "central_velocity": False,
                "test_endpoint_goals": False,
                "test_threshold_tuning": False,
                "internal_val_from_train_only": True,
                "test_sources_unchanged": True,
                "source_overlap_pass": True,
            },
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = hq._gate(payload)
    assert gate["verdict"] == "stage42_hq_group_consistency_weak_slice_repair_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_test_threshold_tuning_is_used() -> None:
    metric = {
        "rows": 10,
        "all_improvement": 0.2,
        "t50_improvement": 0.1,
        "t100_raw_frame_diagnostic_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.1,
    }
    payload = {
        "hp_input": {"exists": True},
        "hp_ucy_before": {"ade_all_improvement": 0.0},
        "ucy_supported_repair": {
            "source": "fresh_ucy_internal_validation_supported_repair",
            "test": {
                "metric": metric,
                "by_domain": {"UCY": metric},
                "diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01},
                "t100_easy_rows": 5,
            },
            "no_leakage": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "central_velocity": False,
                "test_endpoint_goals": False,
                "test_threshold_tuning": True,
                "internal_val_from_train_only": True,
                "test_sources_unchanged": True,
                "source_overlap_pass": True,
            },
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = hq._gate(payload)
    assert gate["gates"]["no_test_threshold_tuning"] is False
    assert gate["passed"] == gate["total"] - 1
