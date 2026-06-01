from __future__ import annotations

from src import stage43_locked_candidate_paper_package_refresh as bi


def _artifact_gate(verdict: str = "ok") -> dict:
    return {"gate": {"verdict": verdict, "passed": 1, "total": 1}}


def _artifacts() -> dict:
    return {
        "candidate_lock": {"stage43_bh_gate": {"verdict": "stage43_bh_protected_multimodal_latent_candidate_lock_pass", "passed": 1, "total": 1}},
        "legacy_paper_refresh": {"stage43_ap_gate": {"verdict": "stage43_ap_paper_evidence_refresh_pass", "passed": 1, "total": 1}},
        "multimodal_head_suite": {"stage43_y_gate": {"verdict": "stage43_y_multimodal_latent_head_suite_pass", "passed": 1, "total": 1}},
        "external_validation_matrix": {"stage43_at_gate": {"verdict": "stage43_at_external_validation_matrix_pass", "passed": 1, "total": 1}},
        "feature_family_multiseed_confirmation": {"stage43_ai_gate": {"verdict": "stage43_ai_feature_family_multiseed_confirmation_pass", "passed": 1, "total": 1}},
        "blocked_source_terms_validation": {"stage43_bg_gate": {"verdict": "stage43_bg_blocked_source_terms_validation_pass", "passed": 1, "total": 1}},
    }


def _payload(*, overclaim: bool = False, latest_positive: bool = True) -> dict:
    value = 0.1 if latest_positive else 0.0
    return {
        "source": bi.SOURCE,
        "input_verdicts": {"candidate_lock": "stage43_bh_protected_multimodal_latent_candidate_lock_pass"},
        "current_claim": {"allowed": [], "disallowed": []},
        "metrics": {
            "rows": 10,
            "all": value,
            "t50": value,
            "t100_raw_frame_diagnostic": 0.0,
            "hard_failure": value,
            "easy_degradation": 0.0,
            "switch_rate": 0.5,
        },
        "evidence": {
            "protected_candidate_locked": True,
            "standalone_world_model_deployable": False,
            "safety_floor_required": True,
            "deployable_proxy_heads": ["a", "b", "c", "d", "e"],
            "diagnostic_only_heads": [],
            "stable_positive_t50_ablation_variants": ["history", "domain"],
            "external_domains": ["ETH_UCY", "TrajNet", "UCY"],
            "source_level_test_rows": 10,
        },
        "source_guard": {
            "ready_for_guarded_conversion_preflight_rows": 0,
            "training_allowed_now": 0,
            "blocked_rows": [],
        },
        "package_outputs": {
            "main_report": "a",
            "claim_boundary": "b",
            "model_card": "c",
            "data_card": "d",
            "reproducibility": "e",
            "a_journal_gap": "f",
            "gate": "g",
        },
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
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "standalone_ungated_deployable": False,
            "uniform_positive_external_transfer_claim": False,
            "source_terms_permission_claim": False,
            "converted_external_support_source": False,
            "a_journal_candidate_now": bool(overclaim),
            "long_objective_complete": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_locked_candidate_package_payload() -> None:
    gate = bi._gate(_payload(overclaim=False, latest_positive=True), _artifacts())
    assert gate["verdict"] == "stage43_bi_locked_candidate_paper_package_refresh_pass"
    assert gate["paper_package_refreshed"] is True
    assert gate["standalone_world_model_deployable"] is False


def test_gate_fails_if_claim_boundary_overstates_a_journal_readiness() -> None:
    gate = bi._gate(_payload(overclaim=True, latest_positive=True), _artifacts())
    assert gate["gates"]["claim_boundary_not_overstated"] is False
    assert gate["paper_package_refreshed"] is False


def test_gate_fails_if_latest_candidate_has_no_positive_lift() -> None:
    gate = bi._gate(_payload(overclaim=False, latest_positive=False), _artifacts())
    assert gate["gates"]["latest_candidate_positive_easy_safe"] is False
    assert gate["protected_multimodal_latent_state_candidate"] is False
