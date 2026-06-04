from __future__ import annotations

from src import stage43_latent_world_state_current_reconciliation as dj


def _payload(
    *,
    deploy_t100: bool = False,
    hide_floor: bool = False,
    metric_claim: bool = False,
    weak_proxy: bool = False,
) -> dict:
    return {
        "source": dj.SOURCE,
        "input_verdicts": {
            "protected_latent_eval": "stage43_c_protected_latent_state_candidate_pass",
            "multimodal_latent_head_suite": "stage43_y_protected_multimodal_latent_head_suite_candidate",
            "protected_multimodal_latent_candidate_lock": "stage43_bh_protected_multimodal_latent_candidate_lock_pass",
            "t100_support_head": "stage43_di_t100_support_aware_distilled_head_safe_but_no_lift_diagnostic",
        },
        "precondition_pass": {
            "protected_latent_eval": True,
            "multimodal_latent_head_suite": True,
            "protected_multimodal_latent_candidate_lock": True,
            "t100_support_head": True,
        },
        "summary": {
            "protected_latent_metrics": {
                "all": 0.17,
                "t50": 0.13,
                "t100_raw_frame_diagnostic": 0.018,
                "hard_failure": 0.18,
                "easy_degradation": 0.0,
                "switch_rate": 0.17,
            },
            "latent_state": {"min_variance": 0.10, "noncollapse_threshold": 0.01},
            "proxy_heads": {
                "failure_risk_auroc": 0.86 if not weak_proxy else 0.51,
                "gain_opportunity_auroc": 0.87,
                "harm_guard_auroc": 0.90,
                "causal_history_density_r2": 0.81,
                "future_interaction_risk_auroc": 0.76,
            },
            "t100_support_head_diagnostic": {
                "mean_t100": 0.0015,
                "mean_min_without_group_t100": 0.0008,
                "all_min_without_group_positive": True,
                "max_easy_degradation": 0.0,
                "beats_dh_t100_mean": False,
                "beats_de_t100_mean": False,
                "deploy_on_current_heldout": deploy_t100,
            },
            "current_deployment_boundary": {
                "safety_floor_required": not hide_floor,
                "standalone_ungated_world_model": False,
                "t100_support_head_deployed": False,
                "stage5c_executed": False,
                "smc_enabled": False,
                "long_objective_complete": False,
            },
        },
        "no_leakage": {
            "protected_latent_eval": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "future_labels_eval_or_loss_only": True,
                "central_velocity_input": False,
                "test_endpoint_goal_construction": False,
                "test_statistics_normalization": False,
            },
            "multimodal_latent_head_suite": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "future_labels_eval_or_supervision_only": True,
                "central_velocity_input": False,
                "test_endpoint_goal_construction": False,
                "test_statistics_normalization": False,
            },
            "t100_support_head": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "future_waypoint_label_eval_only": True,
                "central_velocity_input": False,
                "test_endpoint_goal_construction": False,
                "test_statistics_normalization": False,
            },
        },
        "claim_boundary": {
            "protected_latent_eval": {
                "true_3d": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
            },
            "multimodal_latent_head_suite": {
                "true_3d": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
            },
            "t100_support_head": {
                "true_3d_world_model": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
            },
            "current_public_claim": {
                "true_3d_world_model": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": metric_claim,
                "dataset_local_raw_frame_only": not metric_claim,
                "standalone_ungated_world_model": False,
                "t100_seconds_level_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
                "long_objective_complete": False,
            },
        },
    }


def test_gate_passes_for_protected_latent_current_state() -> None:
    gate = dj._gate(_payload())
    assert gate["verdict"] == "stage43_dj_latent_world_state_current_reconciliation_pass"
    assert gate["protected_multimodal_latent_state_candidate"] is True
    assert gate["standalone_world_model_deployable"] is False
    assert gate["t100_support_head_deployed"] is False


def test_gate_fails_if_t100_diagnostic_is_deployed() -> None:
    gate = dj._gate(_payload(deploy_t100=True))
    assert gate["gates"]["t100_support_head_passed_but_diagnostic"] is False
    assert gate["protected_multimodal_latent_state_candidate"] is False


def test_gate_fails_if_safety_floor_is_hidden() -> None:
    gate = dj._gate(_payload(hide_floor=True))
    assert gate["gates"]["safety_floor_required_not_hidden"] is False
    assert gate["protected_multimodal_latent_state_candidate"] is False


def test_gate_fails_if_metric_or_seconds_claim_is_made() -> None:
    gate = dj._gate(_payload(metric_claim=True))
    assert gate["gates"]["claim_boundary_not_overstated"] is False
    assert gate["protected_multimodal_latent_state_candidate"] is False


def test_gate_fails_if_proxy_heads_are_weak() -> None:
    gate = dj._gate(_payload(weak_proxy=True))
    assert gate["gates"]["proxy_heads_strong_enough"] is False
    assert gate["protected_multimodal_latent_state_candidate"] is False
