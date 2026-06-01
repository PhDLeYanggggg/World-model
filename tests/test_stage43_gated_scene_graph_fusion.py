from __future__ import annotations

import torch

from src import stage43_gated_scene_graph_fusion as bq


def test_gated_scene_graph_model_forward_shapes() -> None:
    model = bq.GatedSceneGraphLatentDynamics(base_dim=6, scene_dim=3, graph_dim=4, hidden_dim=16, latent_dim=8)
    x = torch.zeros(5, 13)
    target = torch.zeros(5, 14)
    out = model(x, target)
    assert out["waypoint_delta"].shape == (5, 4, 2)
    assert out["failure_logit"].shape == (5,)
    assert out["gain_logit"].shape == (5,)
    assert out["harm_logit"].shape == (5,)
    assert out["density"].shape == (5,)
    assert out["z_next"].shape == (5, 8)
    assert out["target_latent"].shape == (5, 8)
    assert out["context_gate"].shape == (5, 2)
    assert torch.all(out["context_gate"] >= 0.0)
    assert torch.all(out["context_gate"] <= 1.0)


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


def _payload(*, all_: float, t50: float, hard: float, easy: float = 0.0) -> dict:
    gated = _metrics(all_, t50, hard, easy)
    best = _metrics(0.12, 0.14, 0.13, 0.0)
    no_context = _metrics(0.10, 0.10, 0.10, 0.0)
    return {
        "source": bq.SOURCE,
        "result_source": "fresh_gated_scene_graph_latent_fusion",
        "model": {
            "dims": {"base_dim": 20, "scene_dim": 4, "graph_dim": 5},
            "latent_variance": 0.2,
            "checkpoint_committed": False,
            "test_metrics_with_floor": gated,
            "gate_summary": {
                "scene_gate_mean": 0.2,
                "graph_gate_mean": 0.3,
                "scene_gate_easy_mean": 0.1,
                "graph_gate_easy_mean": 0.1,
                "scene_gate_hard_mean": 0.4,
                "graph_gate_hard_mean": 0.5,
            },
        },
        "bp_precondition": {
            "verdict": "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic",
            "best_single_by_t50": "graph_history_only",
        },
        "best_single_metrics": best,
        "no_context_metrics": no_context,
        "gated_minus_best_single_by_t50": bq._metric_delta(gated, best),
        "gated_minus_no_context": bq._metric_delta(gated, no_context),
        "ablation_type": {
            "fresh_retrained_gated_fusion": True,
            "not_raw_concat": True,
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


def test_gate_marks_contribution_when_gated_fusion_beats_best_single_safely() -> None:
    gate = bq._gate(_payload(all_=0.15, t50=0.16, hard=0.14))
    assert gate["passed"] == gate["total"]
    assert gate["beats_best_single"] is True
    assert gate["full_multimodal_unsafe"] is False
    assert gate["verdict"] == "stage43_bq_gated_scene_graph_fusion_pass_contribution_supported"


def test_gate_marks_safe_no_lift_diagnostic() -> None:
    gate = bq._gate(_payload(all_=0.11, t50=0.11, hard=0.11))
    assert gate["passed"] == gate["total"]
    assert gate["beats_best_single"] is False
    assert gate["beats_no_context"] is True
    assert gate["full_multimodal_unsafe"] is False
    assert gate["verdict"] == "stage43_bq_gated_scene_graph_fusion_pass_safe_no_best_single_lift_diagnostic"


def test_gate_marks_unsafe_diagnostic() -> None:
    gate = bq._gate(_payload(all_=0.15, t50=0.16, hard=0.14, easy=0.08))
    assert gate["passed"] == gate["total"]
    assert gate["full_multimodal_unsafe"] is True
    assert gate["verdict"] == "stage43_bq_gated_scene_graph_fusion_pass_unsafe_diagnostic"
