from src import stage42_unified_row_cache_stress as s42ae


def test_stage42ae_leave_one_domain_weighted_mean() -> None:
    domains = {
        "A": {"rows": 2, "ade_all": {"mean": 0.2}},
        "B": {"rows": 6, "ade_all": {"mean": 0.6}},
        "C": {"rows": 2, "ade_all": {"mean": 1.0}},
    }
    result = s42ae.build_leave_one_domain(domains)
    held_a = next(row for row in result["rows"] if row["held_out_domain"] == "A")
    assert abs(held_a["ade_all"] - 0.7) < 1e-9


def test_stage42ae_finds_weak_domain_and_horizon() -> None:
    stage42x = {
        "stress": {
            "by_domain": {
                "D1": {
                    "rows": 10,
                    "ade_all": {"mean": 0.1, "ci_low": 0.01, "ci_high": 0.2},
                    "ade_t50": {"mean": 0.1, "ci_low": -0.01, "ci_high": 0.2},
                    "ade_hard_failure": {"mean": 0.1, "ci_low": 0.01, "ci_high": 0.2},
                    "ade_easy_degradation": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.01},
                    "fde_t50": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                }
            },
            "by_horizon": {
                "25": {
                    "rows": 5,
                    "ade_all": {"mean": -0.01, "ci_low": -0.02, "ci_high": 0.01},
                    "ade_hard_failure": {"mean": -0.01, "ci_low": -0.02, "ci_high": 0.01},
                    "switch_rate": {"mean": 0.1},
                }
            },
        }
    }
    findings = s42ae.build_stress_findings(stage42x)
    assert findings["weak_domains"][0]["domain"] == "D1"
    assert findings["weak_horizons"][0]["horizon"] == "25"
    assert findings["limitations"]


def test_stage42ae_gate_passes_with_limitations() -> None:
    stage42x = {
        "summary": {
            "ade_t50": {"ci_low": 0.01},
            "ade_easy_degradation": {"ci_high": 0.01},
        },
        "bootstrap_seed_mean": {"t50": {"ci_low": 0.01}},
        "stage42_x_gate": {"verdict": "stage42_x_unified_row_level_full_waypoint_cache_pass"},
        "no_leakage": {"future_endpoint_input": False},
        "stress": {
            "by_domain": {
                "A": {
                    "rows": 10,
                    "ade_all": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                    "ade_t50": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                    "ade_hard_failure": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                    "ade_easy_degradation": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.01},
                    "fde_t50": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                },
                "B": {
                    "rows": 10,
                    "ade_all": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                    "ade_t50": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                    "ade_hard_failure": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                    "ade_easy_degradation": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.01},
                    "fde_t50": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                },
                "C": {
                    "rows": 10,
                    "ade_all": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                    "ade_t50": {"mean": 0.1, "ci_low": -0.01, "ci_high": 0.2},
                    "ade_hard_failure": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                    "ade_easy_degradation": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.01},
                    "fde_t50": {"mean": 0.1, "ci_low": 0.02, "ci_high": 0.2},
                },
            },
            "by_horizon": {
                "25": {
                    "rows": 5,
                    "ade_all": {"mean": -0.01, "ci_low": -0.02, "ci_high": 0.01},
                    "ade_hard_failure": {"mean": -0.01, "ci_low": -0.02, "ci_high": 0.01},
                    "switch_rate": {"mean": 0.1},
                }
            },
        },
    }
    payload = {
        "stage42x": stage42x,
        "stress_findings": s42ae.build_stress_findings(stage42x),
        "leave_one_domain": s42ae.build_leave_one_domain(s42ae._domain_rows(stage42x)),
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42ae._gate(payload)
    assert gate["passed"] == gate["total"]
