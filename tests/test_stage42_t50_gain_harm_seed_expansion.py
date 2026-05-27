from src.stage42_t50_gain_harm_seed_expansion import _gate, _seed_table, _selected_by_val


def _row(seed, val_t50, test_t50, easy=0.0):
    return {
        "source": "fresh_run",
        "seed": seed,
        "base_seed": seed + 100,
        "selector_info": {"source": "fresh_run"},
        "val_metrics": {
            "ade": {
                "all_improvement": val_t50 / 2,
                "t50_improvement": val_t50,
                "hard_failure_improvement": val_t50 / 3,
                "t100_improvement": 0.0,
                "easy_degradation": easy,
            }
        },
        "test_metrics": {
            "ade": {
                "all_improvement": test_t50 / 2,
                "t50_improvement": test_t50,
                "t100_improvement": 0.0,
                "hard_failure_improvement": test_t50 / 3,
                "easy_degradation": easy,
                "switch_rate": 0.1,
            },
            "fde": {"t50_improvement": test_t50 * 2},
        },
    }


def test_selected_by_val_uses_validation_not_test():
    rows = [_row(1, 0.01, 0.20), _row(2, 0.05, -0.10)]
    selected = _selected_by_val(rows)
    assert selected["seed"] == 2
    assert selected["test_ade_t50"] == -0.10


def test_seed_table_contains_selector_source_and_metrics():
    table = _seed_table([_row(5, 0.02, 0.03)])
    assert table[0]["seed"] == 5
    assert table[0]["selector_source"] == "fresh_run"
    assert table[0]["ade_t50"] == 0.03


def test_gate_passes_when_expanded_seed_ci_positive():
    payload = {
        "original_seed_count": 3,
        "extra_seed_count": 3,
        "combined_seed_count": 6,
        "expanded_summary": {
            "ade_all": {"mean": 0.05},
            "ade_t50": {"mean": 0.03, "ci_low": 0.01},
            "fde_t50": {"mean": 0.06, "ci_low": 0.02},
            "ade_hard_failure": {"mean": 0.04},
            "ade_easy_degradation": {"mean": 0.01},
        },
        "validation_selected": {"test_ade_t50": 0.03},
        "expanded_domain_instability": {},
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
    assert gate["verdict"] == "stage42_ih_t50_seed_expansion_stabilizes_ade_t50"
    assert gate["passed"] == gate["total"]


def test_gate_flags_positive_mean_negative_ci_blocker():
    payload = {
        "original_seed_count": 3,
        "extra_seed_count": 3,
        "combined_seed_count": 6,
        "expanded_summary": {
            "ade_all": {"mean": 0.05},
            "ade_t50": {"mean": 0.03, "ci_low": -0.01},
            "fde_t50": {"mean": 0.06, "ci_low": 0.02},
            "ade_hard_failure": {"mean": 0.04},
            "ade_easy_degradation": {"mean": 0.01},
        },
        "validation_selected": {"test_ade_t50": 0.03},
        "expanded_domain_instability": {},
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
    assert gate["gates"]["combined_ade_t50_seed_ci_positive"] is False
    assert gate["verdict"] == "stage42_ih_t50_seed_expansion_mean_positive_ci_blocker_remains"
