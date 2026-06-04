from __future__ import annotations

import torch

from src import stage43_graph_first_scene_residual_moe as dl


def test_graph_first_scene_residual_moe_forward_shapes() -> None:
    model = dl.GraphFirstSceneResidualMoE(base_dim=6, scene_dim=3, graph_dim=4, hidden_dim=16, latent_dim=8)
    x = torch.zeros(5, 13)
    target = torch.zeros(5, 14)
    out = model(x, target)
    assert out["waypoint_delta"].shape == (5, 4, 2)
    assert out["graph_waypoint_delta"].shape == (5, 4, 2)
    assert out["failure_logit"].shape == (5,)
    assert out["gain_logit"].shape == (5,)
    assert out["harm_logit"].shape == (5,)
    assert out["density"].shape == (5,)
    assert out["z_next"].shape == (5, 8)
    assert out["z_next_graph"].shape == (5, 8)
    assert out["target_latent"].shape == (5, 8)
    assert out["scene_gate"].shape == (5,)
    assert torch.all(out["scene_gate"] >= 0.0)
    assert torch.all(out["scene_gate"] <= 1.0)


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
    current = _metrics(all_, t50, hard, easy)
    best = _metrics(0.12, 0.14, 0.13, 0.0)
    bq = _metrics(0.08, 0.04, 0.07, 0.0)
    no_context = _metrics(0.05, 0.05, 0.05, 0.0)
    return {
        "source": dl.SOURCE,
        "result_source": "fresh_graph_first_scene_residual_moe",
        "dk_precondition": {
            "next_training_contract": "stage43_next_graph_first_scene_residual_moe",
        },
        "model": {
            "dims": {"base_dim": 20, "scene_dim": 4, "graph_dim": 5},
            "latent_variance": 0.2,
            "checkpoint_committed": False,
            "test_metrics_with_floor": current,
            "training_history": [{"epoch": 1, "graph_preservation": 0.1}],
            "scene_residual_gate_summary": {
                "mean": 0.2,
                "easy_mean": 0.05,
                "hard_failure_mean": 0.4,
                "t50_mean": 0.3,
            },
        },
        "ablation_type": {
            "fresh_retrained_graph_first_scene_residual_moe": True,
            "graph_default_expert": True,
            "scene_residual_expert": True,
            "expert_preservation_loss": True,
        },
        "moe_minus_best_single_by_t50": dl.bp._metric_delta(current, best),
        "moe_minus_bq_gated_fusion": dl.bp._metric_delta(current, bq),
        "moe_minus_no_context": dl.bp._metric_delta(current, no_context),
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


def test_gate_marks_contribution_when_moe_beats_best_single_safely() -> None:
    gate = dl._gate(_payload(all_=0.15, t50=0.16, hard=0.14))
    assert gate["passed"] == gate["total"]
    assert gate["beats_best_single"] is True
    assert gate["beats_bq_gated_fusion"] is True
    assert gate["safe_easy"] is True
    assert gate["verdict"] == "stage43_dl_graph_first_scene_residual_moe_pass_contribution_supported"


def test_gate_marks_safe_bq_lift_diagnostic_without_best_single_lift() -> None:
    gate = dl._gate(_payload(all_=0.10, t50=0.10, hard=0.10))
    assert gate["passed"] == gate["total"]
    assert gate["beats_best_single"] is False
    assert gate["beats_bq_gated_fusion"] is True
    assert gate["verdict"] == "stage43_dl_graph_first_scene_residual_moe_pass_safe_bq_lift_diagnostic"


def test_gate_marks_unsafe_diagnostic() -> None:
    gate = dl._gate(_payload(all_=0.15, t50=0.16, hard=0.14, easy=0.08))
    assert gate["passed"] == gate["total"]
    assert gate["safe_easy"] is False
    assert gate["verdict"] == "stage43_dl_graph_first_scene_residual_moe_pass_unsafe_diagnostic"
