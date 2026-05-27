from __future__ import annotations

from src import stage42_fh_source_robustness_audit as fj


def _payload() -> dict:
    return {
        "source": fj.SOURCE,
        "fh_policy": {"fi_verdict": "stage42_fi_fh_policy_freeze_replay_pass"},
        "summary": {
            "domain_count": 2,
            "source_count": 3,
            "domain_horizon_count": 6,
            "robust_domains": ["UCY", "TrajNet"],
            "weak_domains": [],
            "weak_domain_horizons": ["TrajNet|100"],
            "weak_sources": ["TrajNet/Train/crowds/crowds_zara03.txt"],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "internal_val_from_train_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "dual_domain_positive_safe_claim": True,
            "broad_uniform_source_claim": False,
            "broad_uniform_horizon_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_when_weak_slices_are_reported_without_overclaim() -> None:
    gate = fj._gate(_payload())

    assert gate["verdict"] == "stage42_fj_fh_source_robustness_pass"
    assert gate["gates"]["at_least_two_robust_domains"] is True
    assert gate["gates"]["broad_source_claim_only_if_no_weak_sources"] is True
    assert gate["gates"]["broad_horizon_claim_only_if_no_weak_horizons"] is True


def test_gate_blocks_broad_source_overclaim() -> None:
    payload = _payload()
    payload["claim_boundary"]["broad_uniform_source_claim"] = True

    gate = fj._gate(payload)

    assert gate["verdict"] == "stage42_fj_fh_source_robustness_partial"
    assert gate["gates"]["broad_source_claim_only_if_no_weak_sources"] is False


def test_gate_blocks_broad_horizon_overclaim() -> None:
    payload = _payload()
    payload["claim_boundary"]["broad_uniform_horizon_claim"] = True

    gate = fj._gate(payload)

    assert gate["verdict"] == "stage42_fj_fh_source_robustness_partial"
    assert gate["gates"]["broad_horizon_claim_only_if_no_weak_horizons"] is False


def test_summary_section_records_claim_boundaries() -> None:
    payload = _payload()
    payload["stage42_fj_gate"] = fj._gate(payload)
    payload["summary"].update(
        {
            "robust_domain_horizons": ["UCY|50"],
            "robust_sources": ["UCY/UCY/zara01/crowds_zara01.txt"],
            "dual_domain_positive_safe_claim_allowed": True,
            "broad_uniform_source_claim_allowed": False,
            "broad_uniform_horizon_claim_allowed": False,
        }
    )

    section = fj._summary_section(payload)

    assert "Stage42-FJ FH Source" in section
    assert "broad uniform source claim allowed" in section
    assert "False" in section
