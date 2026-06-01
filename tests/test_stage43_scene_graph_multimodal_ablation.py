from __future__ import annotations

import numpy as np

from src import stage43_scene_graph_multimodal_ablation as bp


def test_scene_and_graph_feature_matrices_align_rows() -> None:
    ids = np.arange(24)
    scene_x, scene_names, scene_summary = bp._scene_feature_matrix("test", ids, include_scene=True)
    graph_x, graph_names, graph_summary = bp.bo._graph_feature_matrix(
        "test", ids, include_current=True, include_history=True
    )
    assert scene_x.shape[0] == len(ids)
    assert graph_x.shape[0] == len(ids)
    assert scene_x.shape[1] == len(scene_names)
    assert graph_x.shape[1] == len(graph_names)
    assert scene_summary["scene_feature_count"] > 0
    assert graph_summary["graph_feature_count"] > 0
    assert np.all(np.isfinite(scene_x))
    assert np.all(np.isfinite(graph_x))


def _metrics(all_: float, t50: float, hard: float, easy: float = 0.0) -> dict:
    return {
        "rows": 100,
        "full_waypoint_ade_improvement_vs_floor": all_,
        "endpoint_fde_improvement_vs_floor": all_,
        "t50_full_waypoint_ade_improvement_vs_floor": t50,
        "t50_endpoint_fde_improvement_vs_floor": t50,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": hard,
        "easy_degradation_vs_floor": easy,
        "switch_rate": 0.2,
    }


def _row(
    name: str,
    scene_features: int,
    graph_features: int,
    all_: float,
    t50: float,
    hard: float,
    easy: float = 0.0,
) -> dict:
    return {
        "variant": name,
        "context_feature_count": scene_features + graph_features,
        "scene_feature_count": scene_features,
        "graph_feature_count": graph_features,
        "checkpoint_committed": False,
        "latent_variance": 0.2,
        "test_metrics_with_floor": _metrics(all_, t50, hard, easy=easy),
    }


def _payload(*, full_lifts_best: bool, full_easy_degradation: float = 0.0) -> dict:
    no_context = _row("no_context", 0, 0, 0.10, 0.10, 0.10)
    scene = _row("scene_proxy_only", 14, 0, 0.12, 0.12, 0.11)
    graph = _row("graph_history_only", 0, 17, 0.14, 0.14, 0.13)
    if full_lifts_best:
        full = _row("scene_graph_full", 14, 17, 0.18, 0.19, 0.17, easy=full_easy_degradation)
    else:
        full = _row("scene_graph_full", 14, 17, 0.11, 0.11, 0.105, easy=full_easy_degradation)
    rows = [no_context, scene, graph, full]
    for row in rows:
        row["scene_graph_full_minus_variant"] = bp._metric_delta(
            full["test_metrics_with_floor"], row["test_metrics_with_floor"]
        )
    return {
        "source": bp.SOURCE,
        "result_source": "fresh_retrained_scene_graph_multimodal_ablation",
        "variants": rows,
        "best_single_by_t50": "graph_history_only",
        "scene_graph_full_minus_no_context": bp._metric_delta(
            full["test_metrics_with_floor"], no_context["test_metrics_with_floor"]
        ),
        "scene_graph_full_minus_scene_proxy_only": bp._metric_delta(
            full["test_metrics_with_floor"], scene["test_metrics_with_floor"]
        ),
        "scene_graph_full_minus_graph_history_only": bp._metric_delta(
            full["test_metrics_with_floor"], graph["test_metrics_with_floor"]
        ),
        "scene_graph_full_minus_best_single_by_t50": bp._metric_delta(
            full["test_metrics_with_floor"], graph["test_metrics_with_floor"]
        ),
        "bootstrap_multimodal_vs_no_context_ci": {"n": 500},
        "bootstrap_multimodal_vs_best_single_t50_ci": {"n": 500},
        "preconditions": {
            "stage43_aa_verdict": "stage43_aa_scene_raster_proxy_tokens_pass",
            "stage43_ag_verdict": "stage43_ag_scene_proxy_retrained_ablation_pass",
            "stage43_bo_verdict": "stage43_bo_graph_history_retrained_ablation_pass_contribution_supported",
        },
        "ablation_type": {
            "not_inference_masking": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }


def test_gate_marks_supported_when_full_lifts_best_single() -> None:
    gate = bp._gate(_payload(full_lifts_best=True))
    assert gate["passed"] == gate["total"]
    assert gate["multimodal_contribution_supported"] is True
    assert gate["best_single_lift_supported"] is True
    assert gate["verdict"] == "stage43_bp_scene_graph_multimodal_ablation_pass_contribution_supported"


def test_gate_keeps_mixed_diagnostic_when_full_does_not_lift_best_single() -> None:
    gate = bp._gate(_payload(full_lifts_best=False))
    assert gate["passed"] == gate["total"]
    assert gate["multimodal_contribution_supported"] is True
    assert gate["best_single_lift_supported"] is False
    assert gate["full_multimodal_unsafe"] is False
    assert gate["verdict"] == "stage43_bp_scene_graph_multimodal_ablation_pass_mixed_diagnostic"


def test_gate_marks_completed_negative_diagnostic_when_full_multimodal_is_unsafe() -> None:
    gate = bp._gate(_payload(full_lifts_best=False, full_easy_degradation=0.13))
    assert gate["passed"] == gate["total"]
    assert gate["multimodal_contribution_supported"] is True
    assert gate["best_single_lift_supported"] is False
    assert gate["full_multimodal_unsafe"] is True
    assert gate["verdict"] == "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic"
