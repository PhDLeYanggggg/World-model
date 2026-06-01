from __future__ import annotations

from src.stage43_protected_multimodal_latent_candidate_lock import _gate, _metric, _role_rows


def test_role_rows_indexes_comparison_roles() -> None:
    rows = _role_rows({"comparison_rows": [{"role": "a", "metrics": {"all": 1.0}}, {"role": "b"}]})
    assert rows["a"]["metrics"]["all"] == 1.0
    assert "b" in rows


def test_metric_reads_nested_metrics_as_float() -> None:
    assert _metric({"metrics": {"all": "0.25"}}, "all") == 0.25
    assert _metric({"metrics": {}}, "missing") == 0.0


def test_gate_passes_candidate_lock_payload() -> None:
    artifacts = {
        "safety_floor_replay": {"stage43_a_gate": {"verdict": "stage43_a_safety_floor_replay_pass", "passed": 1, "total": 1}},
        "latent_dataset_contract": {
            "stage43_b_gate": {"verdict": "stage43_b_latent_state_dataset_contract_pass", "passed": 1, "total": 1}
        },
        "protected_latent_eval": {
            "stage43_c_gate": {"verdict": "stage43_c_protected_latent_state_candidate_pass", "passed": 1, "total": 1}
        },
        "full_waypoint_latent_dynamics": {
            "stage43_m_gate": {
                "verdict": "stage43_m_protected_full_waypoint_latent_candidate_pass",
                "passed": 1,
                "total": 1,
            }
        },
        "multimodal_latent_head_suite": {"stage43_y_gate": {"passed": 1, "total": 1}},
        "feature_family_multiseed_confirmation": {"stage43_ai_gate": {"passed": 1, "total": 1}},
        "external_validation_matrix": {"stage43_at_gate": {"passed": 1, "total": 1}},
        "current_candidate_reconciliation": {
            "stage43_ay_gate": {"verdict": "stage43_ay_current_candidate_reconciliation_pass", "passed": 1, "total": 1}
        },
        "blocked_source_terms_validation": {"stage43_bg_gate": {"passed": 1, "total": 1}},
    }
    payload = {
        "input_verdicts": {
            "safety_floor_replay": "stage43_a_safety_floor_replay_pass",
            "latent_dataset_contract": "stage43_b_latent_state_dataset_contract_pass",
            "protected_latent_eval": "stage43_c_protected_latent_state_candidate_pass",
            "full_waypoint_latent_dynamics": "stage43_m_protected_full_waypoint_latent_candidate_pass",
            "current_candidate_reconciliation": "stage43_ay_current_candidate_reconciliation_pass",
        },
        "summary": {
            "protected_multimodal_latent_state_candidate": True,
            "standalone_world_model_deployable": False,
            "safety_floor_required": True,
            "external_domains": ["ETH_UCY", "TrajNet", "UCY"],
            "latest_full_test_tail_adapter_candidate": {
                "deployable": True,
                "all": 0.5,
                "t50": 0.4,
                "hard_failure": 0.3,
                "easy_degradation": 0.0,
            },
            "blocked_source_ready_for_guarded_conversion_preflight_rows": 0,
            "blocked_source_training_allowed_now": 0,
        },
        "ablation_evidence": {"gate_supports_multiseed_module_contribution": True},
        "no_leakage_and_execution": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "new_training_executed": False,
            "new_conversion_executed": False,
        },
        "claim_boundary": {
            "standalone_ungated_deployable": False,
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "source_terms_permission_claim": False,
            "converted_external_support_source": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = _gate(payload, artifacts)
    assert gate["verdict"] == "stage43_bh_protected_multimodal_latent_candidate_lock_pass"
    assert gate["protected_multimodal_latent_state_candidate"] is True
    assert gate["standalone_world_model_deployable"] is False
