import numpy as np

from src import stage42_t50_ensemble_ucy_specialist_integration as ik


def test_metric_rows_tracks_improvement_and_easy_degradation():
    labels = {
        "horizon": np.array([50, 50, 100, 10]),
        "hard": np.array([True, False, False, False]),
        "failure": np.array([False, False, True, False]),
        "easy": np.array([False, True, False, True]),
    }
    floor = np.array([1.0, 1.0, 1.0, 1.0])
    selected = np.array([0.5, 0.9, 0.7, 1.0])
    switch = np.array([True, True, True, False])
    row = ik._metric_rows(selected, floor, labels, switch)
    assert row["t50_improvement"] > 0.0
    assert row["hard_failure_improvement"] > 0.0
    assert row["easy_degradation"] <= 0.02


def test_gate_requires_ucy_repair_and_scope_boundary():
    payload = {
        "source_labels": {
            "stage42ii_non_ucy_policy": "cached_verified_rebuilt_from_stage42ii_intermediates",
            "stage42x_ucy_specialist": "cached_verified_row_aligned_full_waypoint_branch",
            "composition_eval": "fresh_run",
        },
        "alignment": {"ucy_mask_matches_domain": True, "stage42x_ucy_rows": 10, "stage42ii_ucy_rows": 10},
        "summary": {
            "ade_all": 0.1,
            "ade_t50": 0.08,
            "ade_t50_ci_low": 0.02,
            "ucy_t50": 0.03,
            "negative_powered_t50_source_count": 0,
            "ade_hard_failure": 0.1,
            "ade_easy_degradation": 0.0,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "source_specialist_claim_only": True,
            "independent_new_domain_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = ik._gate(payload)
    assert gate["verdict"] == "stage42_ik_ucy_specialist_integration_pass"
    assert gate["passed"] == gate["total"]


def test_stage42_ik_run_keeps_no_overclaim():
    result = ik.run_stage42_t50_ensemble_ucy_specialist_integration()
    gate = result["stage42_ik_gate"]
    assert gate["gates"]["ucy_t50_repaired"]
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
