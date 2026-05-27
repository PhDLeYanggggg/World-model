from src.stage42_t50_gain_harm_stability_audit import (
    _domain_instability,
    _gate,
    _row_level_bootstrap_availability,
    _select_validation_seed,
)


def _row(seed, val_t50, test_t50, domain_t50):
    return {
        "seed": seed,
        "base_seed": seed + 1000,
        "val_metrics": {
            "ade": {
                "all_improvement": val_t50 / 2,
                "t50_improvement": val_t50,
                "t100_improvement": 0.0,
                "hard_failure_improvement": val_t50 / 2,
                "easy_degradation": 0.0,
            }
        },
        "test_metrics": {
            "ade": {
                "all_improvement": test_t50 / 2,
                "t50_improvement": test_t50,
                "hard_failure_improvement": test_t50 / 2,
                "easy_degradation": 0.0,
                "by_domain": {
                    "TrajNet": {
                        "rows": 10,
                        "all_improvement": domain_t50 / 2,
                        "t50_improvement": domain_t50,
                        "hard_failure_improvement": domain_t50 / 2,
                        "easy_degradation": 0.0,
                        "switch_rate": 0.1,
                    }
                },
            },
            "fde": {"t50_improvement": test_t50},
        },
    }


def test_validation_selection_uses_val_metrics_only():
    rows = [_row(1, 0.02, 0.50, 0.01), _row(2, 0.10, -0.10, -0.02)]
    selected = _select_validation_seed(rows)
    assert selected["selected_seed"] == 2
    assert selected["ranked"][0]["test_ade"]["t50_improvement"] == -0.10


def test_domain_instability_finds_negative_t50_slice():
    result = _domain_instability([_row(7, 0.02, 0.01, -0.03)])
    assert result["negative_t50_slice_count"] == 1
    assert result["worst_t50_slices"][0]["domain"] == "TrajNet"


def test_row_level_bootstrap_availability_detects_missing_and_present_arrays():
    assert _row_level_bootstrap_availability({"rows": [{"test_metrics": {"ade": 1.0}}]}) is False
    assert _row_level_bootstrap_availability({"row_errors": [1.0, 2.0]}) is True


def test_gate_marks_ci_blocker_when_ade_t50_ci_is_negative():
    payload = {
        "source": "fresh_stage42_if_t50_gain_harm_stability_audit",
        "stage42p_source": "fresh_run",
        "summary": {
            "seed_count": 3,
            "ade_t50_mean": 0.01,
            "paper_stable_ade_t50_claim_supported": False,
            "paper_stable_fde_t50_claim_supported": True,
            "negative_t50_seed_count": 1,
            "validation_selected_test_ade_t50": 0.02,
        },
        "domain_instability": {"negative_t50_slice_count": 1},
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
    assert gate["gates"]["paper_stable_ade_t50_ci_positive"] is False
    assert gate["verdict"] == "stage42_if_t50_gain_harm_ci_blocker_identified"
