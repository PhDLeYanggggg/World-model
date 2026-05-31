from __future__ import annotations

import numpy as np

from src import stage43_full_waypoint_latent_robustness_audit as n


def test_slice_metrics_reports_improvement_and_easy_harm() -> None:
    floor_ade = np.asarray([1.0, 1.0, 2.0], dtype=np.float32)
    floor_fde = np.asarray([1.0, 1.0, 2.0], dtype=np.float32)
    selected_ade = np.asarray([0.8, 1.2, 1.0], dtype=np.float32)
    selected_fde = np.asarray([0.7, 1.1, 1.0], dtype=np.float32)
    ungated_ade = np.asarray([0.7, 2.0, 1.0], dtype=np.float32)
    switch = np.asarray([True, True, False])
    easy = np.asarray([False, True, False])
    mask = np.asarray([True, True, True])
    out = n._slice_metrics(floor_ade, floor_fde, selected_ade, selected_fde, ungated_ade, switch, easy, mask)
    assert out["rows"] == 3
    assert out["full_waypoint_ade_improvement_vs_floor"] > 0.0
    assert out["easy_degradation_vs_floor"] > 0.0


def test_gate_accepts_full_test_candidate_with_t100_blocker() -> None:
    payload = {
        "source": n.SOURCE,
        "stage43_m_precondition": {"deploy_neural_full_waypoint": True, "sampled_test_rows": 10},
        "checkpoint_sha256_matches_stage43_m": True,
        "full_test_rows": 20,
        "by_domain": {"UCY": {}, "TrajNet": {}},
        "by_horizon": {
            "10": {"full_waypoint_ade_improvement_vs_floor": 0.1},
            "25": {"full_waypoint_ade_improvement_vs_floor": 0.1},
            "50": {"full_waypoint_ade_improvement_vs_floor": 0.1},
            "100": {"full_waypoint_ade_improvement_vs_floor": -0.1},
        },
        "by_source_summary": {"source_count": 2},
        "source_domain_caveats": {
            "uniform_source_success": False,
            "negative_source_count": 1,
            "domain_easy_harm_count": 1,
        },
        "overall_full_test_metrics": {
            "full_waypoint_ade_improvement_vs_floor": 0.2,
            "easy_degradation_vs_floor": 0.0,
        },
        "t100_failure_attribution": {"protected_t100_improvement": -0.1},
        "no_leakage": {"future_waypoint_input": False},
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "t100_success": False,
        },
    }
    gate = n._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["t100_success"] is False
    assert gate["verdict"] == "stage43_n_full_test_positive_with_source_t100_blockers"
