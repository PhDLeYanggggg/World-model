from __future__ import annotations

from src import stage43_t100_family_limited_reconciliation as bk


def _artifact(key: str, verdict: str) -> dict:
    return {key: {"verdict": verdict, "passed": 1, "total": 1}}


def _artifacts() -> dict:
    return {
        "stage43_p_tail_adapter": _artifact(
            "stage43_p_gate", "stage43_p_tail_horizon_adapter_pass_t100_still_fallback"
        ),
        "stage43_t_source_stable_h100_specialist": _artifact(
            "stage43_t_gate", "stage43_t_source_stable_h100_specialist_deployable"
        ),
        "stage43_u_integrated_tail_h100_policy": _artifact(
            "stage43_u_gate", "stage43_u_integrated_tail_h100_policy_pass_family_limited"
        ),
        "stage43_bi_locked_candidate_package": _artifact(
            "stage43_bi_gate", "stage43_bi_locked_candidate_paper_package_refresh_pass"
        ),
        "stage43_bj_long_objective_audit": _artifact(
            "stage43_bj_gate", "stage43_bj_long_objective_evidence_audit_pass_keep_goal_active"
        ),
    }


def _payload(
    *,
    uniform_t100: bool = False,
    endpoint_success: bool = False,
    endpoint_negative_reported: bool = True,
    t100_ci_low: float = 0.001,
    h100_ci_low: float = 0.01,
) -> dict:
    endpoint = -0.005 if endpoint_negative_reported else 0.005
    return {
        "source": bk.SOURCE,
        "input_verdicts": {
            "stage43_p": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
            "stage43_t": "stage43_t_source_stable_h100_specialist_deployable",
            "stage43_u": "stage43_u_integrated_tail_h100_policy_pass_family_limited",
            "stage43_bi": "stage43_bi_locked_candidate_paper_package_refresh_pass",
            "stage43_bj": "stage43_bj_long_objective_evidence_audit_pass_keep_goal_active",
        },
        "stage43_p_reference": {
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0,
        },
        "stage43_t_source_stable_h100": {
            "bootstrap_full_waypoint_ade_ci": {"low": h100_ci_low, "mean": 0.02, "high": 0.03},
        },
        "stage43_u_integrated_policy": {
            "all_full_waypoint_ade_improvement_vs_floor": 0.5,
            "t50_full_waypoint_ade_improvement_vs_floor": 0.51,
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0018,
            "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.48,
            "easy_degradation_vs_floor": 0.0,
            "t100_bootstrap_ci": {"low": t100_ci_low, "mean": 0.0018, "high": 0.0022},
        },
        "h100_source_stable_slice": {
            "integrated_slice": {
                "full_waypoint_ade_improvement_vs_floor": 0.026,
                "endpoint_fde_improvement_vs_floor": endpoint,
            }
        },
        "claim_update": {
            "h100_endpoint_fde_negative_explicitly_reported": endpoint_negative_reported,
            "uniform_t100_success": uniform_t100,
            "t100_endpoint_success": endpoint_success,
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
            "fresh_reconciliation_only": True,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "uniform_t100_success": uniform_t100,
            "t100_endpoint_success": endpoint_success,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }


def test_gate_passes_family_limited_t100_reconciliation() -> None:
    gate = bk._gate(_payload(), _artifacts())
    assert gate["verdict"] == "stage43_bk_t100_family_limited_reconciliation_pass"
    assert gate["t100_family_limited_ade_signal"] is True
    assert gate["uniform_t100_success"] is False
    assert gate["t100_endpoint_success"] is False


def test_gate_fails_if_uniform_t100_is_overclaimed() -> None:
    gate = bk._gate(_payload(uniform_t100=True), _artifacts())
    assert gate["gates"]["uniform_t100_not_overclaimed"] is False
    assert gate["passed"] < gate["total"]


def test_gate_fails_if_endpoint_blocker_is_not_explicit() -> None:
    gate = bk._gate(_payload(endpoint_negative_reported=False, endpoint_success=True), _artifacts())
    assert gate["gates"]["h100_endpoint_blocker_explicit"] is False
    assert gate["passed"] < gate["total"]


def test_gate_fails_if_t100_ci_not_positive() -> None:
    gate = bk._gate(_payload(t100_ci_low=0.0), _artifacts())
    assert gate["gates"]["integrated_t100_ci_positive"] is False
    assert gate["t100_family_limited_ade_signal"] is False


def test_gate_fails_if_h100_source_ci_not_positive() -> None:
    gate = bk._gate(_payload(h100_ci_low=-0.001), _artifacts())
    assert gate["gates"]["h100_source_stable_ade_positive"] is False
    assert gate["t100_family_limited_ade_signal"] is False
