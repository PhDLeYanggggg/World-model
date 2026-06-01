from __future__ import annotations

import numpy as np

from src import stage43_graph_history_retrained_ablation as bo


def test_graph_feature_matrix_empty_for_no_graph() -> None:
    ids = np.arange(16)
    x, names, summary = bo._graph_feature_matrix("test", ids, include_current=False, include_history=False)
    assert x.shape == (16, 0)
    assert names == []
    assert summary["rows"] == 16


def test_graph_feature_matrix_uses_current_and_history_features() -> None:
    ids = np.arange(32)
    x, names, summary = bo._graph_feature_matrix("test", ids, include_current=True, include_history=True)
    assert x.shape[0] == 32
    assert x.shape[1] == len(names)
    assert x.shape[1] >= 10
    assert np.all(np.isfinite(x))
    assert "graph_current_degree" in names
    assert "graph_history_neighbor_valid_degree" in names
    assert summary["graph_feature_count"] == x.shape[1]


def _row(name: str, graph_features: int, all_: float, t50: float, hard: float, easy: float = 0.0) -> dict:
    metrics = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": all_,
        "endpoint_fde_improvement_vs_floor": all_,
        "t50_full_waypoint_ade_improvement_vs_floor": t50,
        "t50_endpoint_fde_improvement_vs_floor": t50,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": hard,
        "easy_degradation_vs_floor": easy,
        "switch_rate": 0.2,
    }
    return {
        "variant": name,
        "graph_feature_count": graph_features,
        "checkpoint_committed": False,
        "latent_variance": 0.2,
        "test_metrics_with_floor": metrics,
    }


def _payload(*, lift: bool) -> dict:
    full = _row("full_graph", 17, 0.13 if lift else 0.10, 0.12 if lift else 0.10, 0.14 if lift else 0.10)
    base = _row("no_graph", 0, 0.10, 0.10, 0.10)
    cur = _row("current_graph_only", 7, 0.11, 0.10, 0.11)
    hist = _row("history_graph_only", 10, 0.11, 0.10, 0.11)
    rows = [base, cur, hist, full]
    for row in rows:
        row["full_graph_minus_variant"] = bo._metric_delta(full["test_metrics_with_floor"], row["test_metrics_with_floor"])
    return {
        "source": bo.SOURCE,
        "result_source": "fresh_retrained_graph_history_ablation",
        "variants": rows,
        "full_graph_minus_no_graph": bo._metric_delta(full["test_metrics_with_floor"], base["test_metrics_with_floor"]),
        "bootstrap_graph_contribution_ci": {"n": 500},
        "preconditions": {"stage43_bn_verdict": "stage43_bn_all_agent_history_graph_cache_pass_raw_scene_blocker"},
        "ablation_type": {"not_inference_masking": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }


def test_gate_marks_supported_when_graph_lifts() -> None:
    gate = bo._gate(_payload(lift=True))
    assert gate["passed"] == gate["total"]
    assert gate["graph_history_retrained_ablation_executed"] is True
    assert gate["graph_history_contribution_supported"] is True
    assert gate["verdict"] == "stage43_bo_graph_history_retrained_ablation_pass_contribution_supported"


def test_gate_keeps_diagnostic_when_no_lift() -> None:
    gate = bo._gate(_payload(lift=False))
    assert gate["passed"] == gate["total"]
    assert gate["graph_history_retrained_ablation_executed"] is True
    assert gate["graph_history_contribution_supported"] is False
    assert gate["verdict"] == "stage43_bo_graph_history_retrained_ablation_pass_diagnostic_no_lift"
