import numpy as np

from src.stage42_t50_gain_harm_row_bootstrap import _bootstrap_ci, _degradation, _gate, _improvement, _selected_seed


def test_improvement_and_degradation_are_ratio_based():
    floor = np.array([10.0, 10.0, 10.0])
    selected = np.array([8.0, 9.0, 10.0])
    ids = np.array([0, 1, 2])
    assert round(_improvement(selected, floor, ids), 6) == round(1.0 - 9.0 / 10.0, 6)
    assert _degradation(selected, floor, ids) == 0.0
    assert round(_degradation(np.array([11.0, 10.0, 10.0]), floor, ids), 6) == round(31.0 / 30.0 - 1.0, 6)


def test_bootstrap_ci_is_positive_for_uniform_gain():
    floor = np.ones(200) * 10.0
    selected = np.ones(200) * 8.0
    mask = np.ones(200, dtype=bool)
    ci = _bootstrap_ci(selected, floor, mask, seed=1, n=100)
    assert ci["low"] > 0.19
    assert ci["high"] < 0.21


def test_selected_seed_uses_validation_score():
    rows = [
        {"seed": 1, "val_metrics": {"ade": {"t50_improvement": 0.02, "all_improvement": 0.01, "hard_failure_improvement": 0.0, "easy_degradation": 0.0}}},
        {"seed": 2, "val_metrics": {"ade": {"t50_improvement": 0.05, "all_improvement": 0.01, "hard_failure_improvement": 0.0, "easy_degradation": 0.0}}},
    ]
    assert _selected_seed(rows) == 2


def test_gate_passes_selected_seed_bootstrap_but_keeps_multiseed_blocker():
    payload = {
        "stage42p_source": "fresh_run",
        "source_labels": {"row_error_replay": "fresh_run", "bootstrap": "fresh_run"},
        "summary": {
            "selected_ade_t50_improvement": 0.03,
            "selected_ade_t50_ci_low": 0.01,
            "selected_fde_t50_ci_low": 0.02,
            "selected_ade_hard_failure_improvement": 0.04,
            "selected_ade_easy_degradation": 0.01,
            "multiseed_ade_t50_ci_low": -0.02,
            "selected_t50_oracle_headroom_ade": 0.20,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["multiseed_ade_t50_ci_still_flagged"] is True
