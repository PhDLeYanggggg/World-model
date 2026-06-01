from __future__ import annotations

from src import stage43_long_objective_evidence_audit as bj


def _artifact(verdict: str, *, key: str) -> dict:
    return {key: {"verdict": verdict, "passed": 1, "total": 1}}


def _artifacts() -> dict:
    return {
        "safety_floor_replay": _artifact("stage43_a_safety_floor_replay_pass", key="stage43_a_gate"),
        "latent_dataset_contract": _artifact("stage43_b_latent_state_dataset_contract_pass", key="stage43_b_gate"),
        "protected_latent_eval": _artifact("stage43_c_protected_latent_state_candidate_pass", key="stage43_c_gate"),
        "data_calibration": _artifact("stage43_as_data_calibration_refresh_pass", key="stage43_as_gate"),
        "external_validation_matrix": _artifact("stage43_at_external_validation_matrix_pass", key="stage43_at_gate"),
        "full_waypoint_latent_dynamics": _artifact("stage43_m_protected_full_waypoint_latent_candidate_pass", key="stage43_m_gate"),
        "feature_family_multiseed": _artifact("stage43_ai_feature_family_multiseed_confirmation_pass", key="stage43_ai_gate"),
        "safety_floor_necessity": _artifact("stage43_aj_safety_floor_necessity_confirmed", key="stage43_aj_gate"),
        "locked_candidate_package": _artifact("stage43_bi_locked_candidate_paper_package_refresh_pass", key="stage43_bi_gate"),
        "blocked_source_terms_validation": _artifact("stage43_bg_blocked_source_terms_validation_pass", key="stage43_bg_gate"),
    }


def _phase(name: str, status: str, complete: bool = False) -> dict:
    return {
        "phase": name,
        "status": status,
        "evidence": "",
        "proved": [],
        "missing": [],
        "next_action": "",
        "pass_for_current_audit": True,
        "complete_for_long_objective": complete,
    }


def _payload(*, overclaim: bool = False, phase_complete: bool = False) -> dict:
    return {
        "source": bj.SOURCE,
        "current_candidate": {
            "metrics": {"t100_raw_frame_diagnostic": 0.0},
        },
        "phases": [
            _phase("A data and calibration", "partial_blocked", phase_complete),
            _phase("B external validation", "pass_with_boundary", phase_complete),
            _phase("C full-waypoint / latent dynamics", "protected_candidate_pass", phase_complete),
            _phase("D causal ablation / module evidence", "partial_supported", phase_complete),
            _phase("E safety floor study", "floor_required", phase_complete),
            _phase("F paper package", "pass_with_a_journal_gap", phase_complete),
        ],
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
            "metric_or_seconds_claim": bool(overclaim),
            "dataset_local_raw_frame_only": True,
            "standalone_ungated_deployable": False,
            "a_journal_candidate_now": False,
            "long_objective_complete": bool(overclaim),
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_audit_while_keeping_long_objective_active() -> None:
    gate = bj._gate(_payload(overclaim=False, phase_complete=False), _artifacts())
    assert gate["verdict"] == "stage43_bj_long_objective_evidence_audit_pass_keep_goal_active"
    assert gate["long_objective_complete"] is False
    assert gate["protected_multimodal_latent_state_candidate"] is True


def test_gate_fails_if_metric_or_completion_is_overclaimed() -> None:
    gate = bj._gate(_payload(overclaim=True, phase_complete=False), _artifacts())
    assert gate["gates"]["claim_boundary_not_overstated"] is False
    assert gate["gates"]["long_objective_kept_active"] is False


def test_gate_fails_if_phases_are_marked_complete_for_long_objective() -> None:
    gate = bj._gate(_payload(overclaim=False, phase_complete=True), _artifacts())
    assert gate["gates"]["long_objective_kept_active"] is False
    assert gate["long_objective_complete"] is False
