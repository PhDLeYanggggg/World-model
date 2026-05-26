from src import stage42_unified_external_full_waypoint_policy as w


def test_stage42_w_ucy_slice_extraction_is_domain_specific():
    report = {
        "best_trial": "best",
        "rows": [
            {
                "trial": {"name": "best"},
                "test_metrics": {
                    "ade": {
                        "by_domain": {
                            "ETH_UCY": {"rows": 20, "all_improvement": -99.0, "t50_improvement": -99.0, "t100_improvement": -99.0, "hard_failure_improvement": -99.0, "easy_degradation": 9.0, "switch_rate": 1.0},
                            "UCY": {"rows": 10, "all_improvement": 0.1, "t50_improvement": 0.2, "t100_improvement": 0.3, "hard_failure_improvement": 0.4, "easy_degradation": 0.0, "switch_rate": 0.5},
                        }
                    },
                    "fde": {
                        "by_domain": {
                            "UCY": {"rows": 10, "t50_improvement": 0.6},
                        }
                    },
                },
            },
            {
                "trial": {"name": "best"},
                "test_metrics": {
                    "ade": {
                        "by_domain": {
                            "UCY": {"rows": 10, "all_improvement": 0.3, "t50_improvement": 0.4, "t100_improvement": 0.5, "hard_failure_improvement": 0.6, "easy_degradation": 0.0, "switch_rate": 0.7},
                        }
                    },
                    "fde": {
                        "by_domain": {
                            "UCY": {"rows": 10, "t50_improvement": 0.8},
                        }
                    },
                },
            },
        ],
    }
    item = w._ucy_domain_from_stage42v(report)
    assert item["rows"] == 10
    assert item["ade_all"]["mean"] == 0.2
    assert item["ade_t50"]["mean"] == 0.30000000000000004
    assert item["fde_t50"]["mean"] == 0.7


def test_stage42_w_weighted_summary_uses_row_counts():
    by_domain = {
        "A": {"rows": 10, "ade_all": {"mean": 0.1, "ci_low": 0.0, "ci_high": 0.2}, "ade_t50": {"mean": 0.2}, "ade_t100_raw_frame_diagnostic": {"mean": 0.0}, "ade_hard_failure": {"mean": 0.0}, "ade_easy_degradation": {"mean": 0.0}, "fde_t50": {"mean": 0.0}, "switch_rate": {"mean": 0.0}},
        "B": {"rows": 30, "ade_all": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}, "ade_t50": {"mean": 0.6}, "ade_t100_raw_frame_diagnostic": {"mean": 0.0}, "ade_hard_failure": {"mean": 0.0}, "ade_easy_degradation": {"mean": 0.0}, "fde_t50": {"mean": 0.0}, "switch_rate": {"mean": 0.0}},
    }
    summary = w._weighted_summary(by_domain)
    assert summary["rows"] == 40
    assert abs(summary["ade_all"]["mean"] - 0.4) < 1e-12
    assert summary["ade_all"]["domain_min_ci_low"] == 0.0


def test_stage42_w_run_outputs_claim_boundaries():
    result = w.run_stage42_unified_external_full_waypoint_policy()
    gate = result["stage42_w_gate"]
    assert gate["gates"]["stage42s_verified"]
    assert gate["gates"]["stage42v_verified"]
    assert gate["gates"]["ucy_replaced_from_stage42v"]
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
