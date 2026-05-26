from src import stage42_source_level_baseline_family_robustness as s42av


def test_ci_positive_handles_gain_and_easy():
    assert s42av._ci_positive({"low": 0.1, "high": 0.2, "bootstrap_n": 100})
    assert not s42av._ci_positive({"low": -0.1, "high": 0.2, "bootstrap_n": 100})
    assert s42av._ci_positive({"low": -0.1, "high": 0.01, "bootstrap_n": 100}, easy=True)
    assert not s42av._ci_positive({"low": -0.1, "high": 0.03, "bootstrap_n": 100}, easy=True)


def test_domain_support_marks_no_validation_floor_only():
    au = {
        "split_stats": {
            "by_split": {
                "train": {"domains": {"UCY": 10}},
                "val": {"domains": {}},
                "test": {"domains": {"UCY": 5}},
            }
        },
        "variants": {
            "baseline_family_all": {
                "by_domain": {
                    "UCY": {
                        "rows": 5,
                        "all_improvement": 0.0,
                        "t50_improvement": 0.0,
                        "easy_degradation": -0.0,
                        "switch_rate": 0.0,
                    }
                }
            }
        },
    }
    out = s42av._domain_support(au, "baseline_family_all")
    assert out["UCY"]["blocker"] == "no_validation_rows_for_domain_policy_selection_floor_only"
    assert out["UCY"]["positive_or_floor_safe"]
    assert not out["UCY"]["positive_transfer"]


def test_horizon_support_flags_easy_degradation():
    au = {
        "variants": {
            "baseline_family_all": {
                "by_horizon": {
                    "100": {
                        "rows": 10,
                        "all_improvement": 0.1,
                        "hard_failure_improvement": 0.1,
                        "easy_degradation": 0.03,
                    }
                }
            }
        }
    }
    out = s42av._horizon_support(au, "baseline_family_all")
    assert "easy_degradation_over_2pct" in out["100"]["weaknesses"]
    assert not out["100"]["positive_and_easy_safe"]


def test_gate_passes_when_global_positive_and_limits_reported():
    result = {
        "au_verdict": "stage42_au_baseline_family_mechanism_pass",
        "global_stability": {
            "baseline_family_all": {
                "all_ci_low_positive": True,
                "t50_ci_low_positive": True,
                "hard_failure_ci_low_positive": True,
                "easy_degradation_ci_high_safe": True,
            }
        },
        "domain_support": {"TrajNet": {}, "UCY": {}},
        "summary": {
            "floor_only_or_blocked_domains": ["UCY"],
            "weak_horizons": ["100"],
            "uniform_domain_claim_allowed": False,
            "uniform_horizon_claim_allowed": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42av._gate(result)
    assert gate["passed"] == gate["total"]
