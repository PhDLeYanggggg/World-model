from __future__ import annotations

from src.stage42_group_consistency_contribution_audit import _gate, _summary


def _dy() -> dict:
    return {
        "stage42_dy_gate": {"passed": 16, "total": 16},
        "summary": {
            "group_consistency_all": 0.247,
            "group_consistency_t50": 0.224,
            "group_consistency_t100_raw_frame_diagnostic": 0.143,
            "group_consistency_hard": 0.239,
            "group_consistency_easy": -0.256,
            "group_consistency_near005_base": 0.019,
            "group_consistency_near005_final": 0.014,
            "best_loss_family_candidate": "proximity_occupancy_loss",
            "best_loss_family_all": 0.255,
            "best_loss_family_t50": 0.221,
            "best_loss_family_hard": 0.237,
            "loss_family_any_promotable_over_stage42_am": False,
        },
    }


def _dz() -> dict:
    return {
        "stage42_dz_gate": {"passed": 15, "total": 15},
        "summary": {
            "positive_safe_domains": 2,
            "ucy_all": 0.356,
            "ucy_t50": 0.227,
            "ucy_hard": 0.338,
            "trajnet_all": 0.321,
            "trajnet_t50": 0.282,
            "trajnet_hard": 0.313,
        },
    }


def _ci(low: float, high: float) -> dict:
    return {"low": low, "mid": (low + high) / 2.0, "high": high}


def _ea() -> dict:
    return {
        "stage42_ea_gate": {"passed": 12, "total": 12},
        "summary": {"bootstrap_n": 2000, "ci_positive_safe_domains": 2},
        "bootstrap_ci": {
            "global": {
                "all": _ci(0.32, 0.33),
                "t50": _ci(0.26, 0.27),
                "hard_failure": _ci(0.31, 0.32),
                "easy_degradation": _ci(-0.33, -0.31),
            },
            "by_domain": {
                "UCY": {
                    "all": _ci(0.34, 0.36),
                    "t50": _ci(0.21, 0.24),
                    "hard_failure": _ci(0.32, 0.34),
                    "easy_degradation": _ci(-0.12, -0.08),
                },
                "TrajNet": {
                    "all": _ci(0.31, 0.32),
                    "t50": _ci(0.27, 0.28),
                    "hard_failure": _ci(0.30, 0.31),
                    "easy_degradation": _ci(-0.28, -0.26),
                },
            },
        },
        "near_collision_ci": {"global": {"delta_final_minus_base": _ci(-0.008, -0.006)}},
    }


def _dp() -> dict:
    return {
        "stage42_dp_gate": {"passed": 19, "total": 19},
        "summary": {
            "baseline_family_metric": {
                "all_improvement": 0.288,
                "t50_improvement": 0.315,
                "hard_failure_improvement": 0.276,
            },
            "positive_context_rows": [],
            "best_delta_all": -0.023,
            "best_delta_t50": -0.083,
            "best_delta_hard_failure": -0.026,
            "root_cause": "context residual target is not extracting independent value",
        },
    }


def test_stage42_ec_summary_separates_supported_and_blocked_claims() -> None:
    summary = _summary(_dy(), _dz(), _ea(), _dp())

    assert summary["supported_contributions"]["explicit_group_consistency_full_waypoint"]["status"] == "supported_source_level"
    assert summary["supported_contributions"]["dual_domain_raw_frame_support"]["positive_safe_domains"] == 2
    assert summary["blocked_or_negative_contributions"]["scalar_loss_family_primary"]["status"] == "blocked"
    assert summary["blocked_or_negative_contributions"]["current_sequence_graph_residual_context"]["status"] == "closed_current_protocol"
    assert summary["blocked_or_negative_contributions"]["ungated_global_full_waypoint_replacement"]["status"] == "blocked"


def test_stage42_ec_gate_passes_when_supported_and_overclaims_blocked() -> None:
    payload = {
        "input_gates": {"dy": True, "dz": True, "ea": True, "dp": True},
        "summary": _summary(_dy(), _dz(), _ea(), _dp()),
        "claim_boundary": {
            "ungated_full_waypoint_deployable": False,
            "global_primary_full_waypoint_replacement_claim_allowed": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ec_group_consistency_contribution_audit_pass"
