from __future__ import annotations

import numpy as np

from src import stage42_t50_floor_relaxability_repair as by


def test_metric_positive_requires_gain_and_easy_safety() -> None:
    good = {
        "all_improvement": 0.05,
        "t50_improvement": 0.08,
        "hard_failure_improvement": 0.11,
        "easy_degradation": 0.01,
        "switch_rate": 0.3,
    }
    assert by._metric_positive(good)

    bad_easy = dict(good, easy_degradation=0.021)
    assert not by._metric_positive(bad_easy)

    no_switch = dict(good, switch_rate=0.0)
    assert not by._metric_positive(no_switch)


def test_domain_horizon_metrics_reports_domain_horizon_slice(monkeypatch) -> None:
    data = {
        "dataset": np.array(["UCY", "UCY", "TrajNet"], dtype=object),
        "horizon": np.array([50, 25, 50]),
    }
    split = np.array(["test", "test", "train"], dtype=object)
    selected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    floor = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    switch = np.array([True, False, True])

    def fake_metric(selected_arg, floor_arg, data_arg, switch_arg, mask_arg):
        assert int(np.sum(mask_arg)) == 1
        return {
            "all_improvement": 0.5,
            "t50_improvement": 0.5,
            "hard_failure_improvement": 0.25,
            "easy_degradation": 0.0,
            "switch_rate": 1.0,
        }

    monkeypatch.setattr(by.am, "_metric", fake_metric)
    out = by._domain_horizon_metrics(data, split, selected, floor, switch)

    assert set(out) == {"UCY|25", "UCY|50"}
    assert out["UCY|50"]["test_rows"] == 1
    assert out["UCY|50"]["metric"]["t50_improvement"] == 0.5


def test_gate_passes_for_protected_t50_repair() -> None:
    payload = {
        "source": "unit",
        "input_reports": {
            "stage42_bx_verdict": "stage42_bx_floor_relaxability_audit_pass",
            "stage42_aw_verdict": "stage42_aw_ucy_validation_support_repair_pass",
            "stage42_bw_verdict": "stage42_bw_safety_floor_necessity_audit_pass",
        },
        "summary": {
            "target_slices": ["TrajNet|50", "UCY|50"],
            "ucy_t50_repaired": True,
            "trajnet_t50_repaired": True,
            "global_t50_improvement": 0.2,
            "global_easy_degradation": 0.0,
            "teacher_floor_context_required": True,
            "floor_free_neural_deployable": False,
        },
        "no_leakage": {
            "internal_val_from_train_only": True,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = by._gate(payload)
    assert gate["verdict"] == "stage42_by_t50_floor_relaxability_repair_pass"
    assert gate["passed"] == gate["total"]


def test_gate_blocks_floor_free_claim() -> None:
    payload = {
        "source": "unit",
        "input_reports": {
            "stage42_bx_verdict": "stage42_bx_floor_relaxability_audit_pass",
            "stage42_aw_verdict": "stage42_aw_ucy_validation_support_repair_pass",
            "stage42_bw_verdict": "stage42_bw_safety_floor_necessity_audit_pass",
        },
        "summary": {
            "target_slices": ["TrajNet|50", "UCY|50"],
            "ucy_t50_repaired": True,
            "trajnet_t50_repaired": True,
            "global_t50_improvement": 0.2,
            "global_easy_degradation": 0.0,
            "teacher_floor_context_required": True,
            "floor_free_neural_deployable": True,
        },
        "no_leakage": {
            "internal_val_from_train_only": True,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "floor_free_neural_deployable": True,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = by._gate(payload)
    assert gate["verdict"] == "stage42_by_t50_floor_relaxability_repair_partial"
    assert not gate["gates"]["not_floor_free_neural"]
