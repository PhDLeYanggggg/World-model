from __future__ import annotations

import numpy as np

from src import stage42_horizon_sequence_graph_context_router as s42io


def test_filter_rows_keeps_row_aligned_arrays_only() -> None:
    data = {
        "horizon": np.asarray([10, 50, 100]),
        "x": np.asarray([[1, 2], [3, 4], [5, 6]]),
        "meta": np.asarray([1, 2]),
        "name": "external",
    }
    mask = np.asarray([True, False, True])
    out = s42io._filter_rows(data, mask)
    assert out["horizon"].tolist() == [10, 100]
    assert out["x"].shape == (2, 2)
    assert out["meta"].shape == (2,)
    assert out["name"] == "external"


def test_positive_horizon_context_requires_gain_and_easy_safety() -> None:
    metric = {
        "all_improvement": 0.011,
        "hard_failure_improvement": 0.0,
        "easy_degradation": 0.0,
    }
    assert s42io._is_positive_horizon_context(metric)
    metric["easy_degradation"] = 0.021
    assert not s42io._is_positive_horizon_context(metric)
    metric["all_improvement"] = 0.0
    metric["hard_failure_improvement"] = 0.009
    metric["easy_degradation"] = 0.0
    assert not s42io._is_positive_horizon_context(metric)


def test_gate_passes_for_bounded_positive_or_negative_horizon_result() -> None:
    metric = {
        "rows": 10,
        "all_improvement": 0.0,
        "t50_improvement": 0.0,
        "t100_raw_frame_diagnostic_improvement": 0.0,
        "hard_failure_improvement": 0.0,
        "easy_degradation": 0.0,
        "switch_rate": 0.0,
    }
    payload = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "baseline_family_control": {"policy_slice_count": 1},
        "sequence_summary_schema": {"stats": {"feature_count": 11}},
        "graph_summary_schema": {"stats": {"rows_with_neighbors": 5}},
        "horizon_routers": {
            f"h{h}_{c}": {
                "validation_selection": {"source": "validation_only", "test_threshold_tuning": False}
            }
            for h in s42io.HORIZONS
            for c in s42io.CANDIDATES
        },
        "best_by_horizon": {str(h): {"metric": metric} for h in s42io.HORIZONS},
        "summary": {
            "horizon_specific_increment_verdict": "stage42_io_horizon_sequence_graph_context_router_not_supported"
        },
        "positive_horizon_sequence_graph_context_routers": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "sequence_summary_current_past_only": True,
            "graph_summary_current_past_only": True,
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
    gate = s42io._gate(payload)
    assert gate["verdict"] == "stage42_io_horizon_sequence_graph_context_router_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_missing_horizon_router() -> None:
    metric = {"rows": 10}
    payload = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "baseline_family_control": {"policy_slice_count": 1},
        "sequence_summary_schema": {"stats": {"feature_count": 11}},
        "graph_summary_schema": {"stats": {"rows_with_neighbors": 5}},
        "horizon_routers": {},
        "best_by_horizon": {str(h): {"metric": metric} for h in s42io.HORIZONS},
        "summary": {
            "horizon_specific_increment_verdict": "stage42_io_horizon_sequence_graph_context_router_not_supported"
        },
        "positive_horizon_sequence_graph_context_routers": [],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "sequence_summary_current_past_only": True,
            "graph_summary_current_past_only": True,
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
    gate = s42io._gate(payload)
    assert gate["passed"] < gate["total"]
    assert not gate["gates"]["horizon_routers_complete"]
