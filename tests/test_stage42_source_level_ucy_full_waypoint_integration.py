from src import stage42_source_level_ucy_full_waypoint_integration as iu


def test_stage42_iu_weighted_summary_uses_rows():
    domains = {
        "A": {
            "rows": 10,
            "ade_all": {"mean": 0.2, "ci_low": 0.1, "ci_high": 0.3},
            "ade_t10": {"mean": 0.0},
            "ade_t25": {"mean": 0.0},
            "ade_t50": {"mean": 0.4},
            "ade_t100_raw_frame_diagnostic": {"mean": 0.0},
            "ade_hard_failure": {"mean": 0.5},
            "ade_easy_degradation": {"mean": 0.0},
            "switch_rate": {"mean": 0.1},
            "harm_over_fallback": {"mean": -0.1},
        },
        "B": {
            "rows": 30,
            "ade_all": {"mean": 0.6, "ci_low": 0.5, "ci_high": 0.7},
            "ade_t10": {"mean": 0.0},
            "ade_t25": {"mean": 0.0},
            "ade_t50": {"mean": 0.8},
            "ade_t100_raw_frame_diagnostic": {"mean": 0.0},
            "ade_hard_failure": {"mean": 0.9},
            "ade_easy_degradation": {"mean": 0.0},
            "switch_rate": {"mean": 0.3},
            "harm_over_fallback": {"mean": -0.2},
        },
    }
    summary = iu._weighted_summary(domains)
    assert summary["rows"] == 40
    assert abs(summary["ade_all"]["mean"] - 0.5) < 1e-12
    assert abs(summary["ade_t50"]["mean"] - 0.7) < 1e-12
    assert summary["ade_all"]["domain_min_ci_low"] == 0.1


def test_stage42_iu_extracts_stage42v_ucy_only():
    report = {
        "best_trial": "best",
        "rows": [
            {
                "trial": {"name": "best"},
                "test_metrics": {
                    "ade": {"by_domain": {"ETH_UCY": {"rows": 99, "all_improvement": -9}, "UCY": {"rows": 10, "all_improvement": 0.2, "t10_improvement": 0.1, "t25_improvement": 0.0, "t50_improvement": 0.4, "t100_improvement": 0.5, "hard_failure_improvement": 0.6, "easy_degradation": 0.0, "switch_rate": 0.3, "harm_over_fallback": -0.1}}},
                    "fde": {"by_domain": {"UCY": {"rows": 10, "t50_improvement": 0.7}}},
                },
            },
            {
                "trial": {"name": "best"},
                "test_metrics": {
                    "ade": {"by_domain": {"UCY": {"rows": 10, "all_improvement": 0.4, "t10_improvement": 0.3, "t25_improvement": 0.0, "t50_improvement": 0.6, "t100_improvement": 0.7, "hard_failure_improvement": 0.8, "easy_degradation": 0.0, "switch_rate": 0.5, "harm_over_fallback": -0.2}}},
                    "fde": {"by_domain": {"UCY": {"rows": 10, "t50_improvement": 0.9}}},
                },
            },
        ],
    }
    item = iu._stage42_v_ucy(report)
    assert item["rows"] == 10
    assert item["domain"] == "UCY"
    assert abs(item["ade_all"]["mean"] - 0.3) < 1e-12
    assert abs(item["fde_t50"]["mean"] - 0.8) < 1e-12


def test_stage42_iu_run_outputs_policy_package_boundaries():
    result = iu.run_stage42_source_level_ucy_full_waypoint_integration()
    gate = result["stage42_iu_gate"]
    assert gate["gates"]["stage42_it_passed"]
    assert gate["gates"]["stage42_v_passed"]
    assert gate["gates"]["ucy_replaced_from_stage42_v"]
    assert gate["gates"]["no_leakage_pass"]
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
