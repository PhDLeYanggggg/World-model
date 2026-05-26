from src import stage42_repaired_protocol_robustness as s42ax


def test_ci_positive_handles_gain_and_easy():
    assert s42ax._ci_positive({"low": 0.01, "high": 0.02, "bootstrap_n": 10})
    assert not s42ax._ci_positive({"low": -0.01, "high": 0.02, "bootstrap_n": 10})
    assert s42ax._ci_positive({"low": -0.5, "high": 0.01, "bootstrap_n": 10}, easy=True)
    assert not s42ax._ci_positive({"low": -0.5, "high": 0.03, "bootstrap_n": 10}, easy=True)


def test_horizon_audit_flags_t100_easy_weakness():
    best = {
        "by_horizon": {
            "100": {
                "rows": 7,
                "all_improvement": 0.1,
                "t100_raw_frame_diagnostic_improvement": 0.1,
                "hard_failure_improvement": 0.1,
                "easy_degradation": 0.023,
                "switch_rate": 0.4,
            }
        }
    }
    out = s42ax._horizon_audit(best)
    assert out["100"]["horizon_metric"] == 0.1
    assert "easy_degradation_over_2pct" in out["100"]["weaknesses"]
    assert not out["100"]["positive_and_easy_safe"]


def test_domain_audit_requires_all_t50_hard_easy_and_switch():
    metric = {
        "rows": 10,
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.5,
    }
    out = s42ax._domain_audit({"by_domain": {"UCY": metric}})
    assert out["UCY"]["all_positive"]
    assert out["UCY"]["t50_positive"]
    assert out["UCY"]["hard_failure_positive"]
    assert out["UCY"]["easy_safe"]
    assert out["UCY"]["switches"]


def test_gate_passes_when_t100_limit_is_reported():
    result = {
        "source": "unit_test",
        "aw_verdict": "stage42_aw_ucy_validation_support_repair_pass",
        "bootstrap_audit": {
            "all_ci_low_positive": True,
            "t50_ci_low_positive": True,
            "t100_raw_frame_diagnostic_ci_low_positive": True,
            "hard_failure_ci_low_positive": True,
            "easy_degradation_ci_high_safe": True,
        },
        "summary": {
            "positive_domains": ["TrajNet", "UCY"],
            "weak_horizons": ["100"],
            "uniform_horizon_claim_allowed": False,
        },
        "before_after": {"blocker_repaired": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "test_sources_unchanged": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42ax._gate(result)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ax_repaired_protocol_robustness_pass_with_t100_limit"
