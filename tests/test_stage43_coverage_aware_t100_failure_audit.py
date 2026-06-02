from __future__ import annotations

from src import stage43_coverage_aware_t100_failure_audit as ch


def _stats(rows: int, improvement: float) -> dict:
    return {
        "rows": rows,
        "full_waypoint_ade_improvement_vs_floor": improvement,
        "endpoint_fde_improvement_vs_floor": improvement,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.5,
        "mean_floor_ade": 1.0,
        "mean_selected_ade": 1.0 - improvement,
        "harm_over_floor_ade": -improvement,
    }


def _payload(t100_improvement: float) -> dict:
    return {
        "coverage_aware_latent_dynamics": {
            "verdict": "stage43_cg_coverage_aware_latent_dynamics_candidate_pass",
            "mode": "medium",
        },
        "horizon_slices": {
            "10": _stats(10, 0.2),
            "25": _stats(10, 0.2),
            "50": _stats(10, 0.2),
            "100": _stats(10, t100_improvement),
        },
        "t100_bootstrap_ci": {"rows": 10, "low": -0.1, "mean": t100_improvement, "high": -0.01},
        "domain_slices": {"UCY": _stats(10, 0.1)},
        "source_slices": {"zara": _stats(10, 0.1)},
        "t100_switch_attribution": {
            "t100_switched": _stats(5, t100_improvement),
            "t100_fallback": _stats(5, 0.0),
        },
        "prediction_diagnostics": {"t100": {"rows": 10}},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_complete_negative_t100_audit() -> None:
    gate = ch._gate(_payload(-0.05))
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_ch_t100_failure_audit_pass_blocker_confirmed"


def test_gate_requires_negative_t100_confirmation() -> None:
    gate = ch._gate(_payload(0.01))
    assert gate["gates"]["t100_negative_confirmed"] is False
    assert gate["verdict"] == "stage43_ch_t100_failure_audit_incomplete"
