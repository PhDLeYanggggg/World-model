from __future__ import annotations

import numpy as np

from src import stage42_common_validation_bridge_shape_composer as co


def _labels() -> dict[str, np.ndarray]:
    return {
        "horizon": np.asarray([50, 50, 100, 100]),
        "hard": np.asarray([True, False, True, False]),
        "failure": np.asarray([False, False, True, False]),
        "easy": np.asarray([False, True, False, True]),
        "domain": np.asarray(["A", "A", "A", "A"]),
        "scene_id": np.asarray(["s", "s", "s", "s"]),
        "source_file": np.asarray(["f", "f", "f", "f"]),
        "current_xy": np.zeros((4, 2), dtype=float),
    }


def test_alignment_report_requires_matching_rows_and_metadata() -> None:
    labels = _labels()
    endpoint = {"labels": labels}
    full = {"labels": {**labels, "current_xy": labels["current_xy"].copy()}}
    out = co._alignment_report(endpoint, full)
    assert out["aligned"] is True
    full["labels"]["horizon"] = np.asarray([50, 25, 100, 100])
    out = co._alignment_report(endpoint, full)
    assert out["aligned"] is False
    assert out["horizon_match"] is False


def test_metric_reports_improvement_and_easy_harm_against_reference() -> None:
    labels = _labels()
    ref = np.asarray([10.0, 10.0, 10.0, 10.0])
    selected = np.asarray([8.0, 10.2, 9.0, 10.0])
    switch = np.asarray([True, True, True, False])
    metric = co._metric(selected, ref, labels, switch)
    assert metric["all_improvement"] > 0.0
    assert metric["t50_improvement"] > 0.0
    assert metric["hard_failure_improvement"] > 0.0
    assert metric["easy_degradation"] > 0.0


def test_gate_passes_for_validation_selected_safe_composer() -> None:
    payload = {
        "alignment": {
            "val": {"aligned": True},
            "test": {"aligned": True},
        },
        "policy_selection": {"selected_on": "validation_only", "test_evaluated_once": True},
        "test_eval": {
            "metric_vs_endpoint_ade": {
                "all_improvement": 0.01,
                "t50_improvement": 0.02,
                "easy_degradation": 0.0,
            },
            "metric_vs_floor_ade": {
                "all_improvement": 0.2,
                "t50_improvement": 0.1,
            },
        },
        "no_leakage": {"test_threshold_tuning": False},
        "paper_file_status": [{"contains_stage42_co": True}],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = co._gate(payload)
    assert gate["verdict"] == "stage42_co_common_validation_bridge_shape_composer_pass"
    assert gate["passed"] == gate["total"]
