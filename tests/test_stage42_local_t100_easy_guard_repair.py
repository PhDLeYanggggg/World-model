from src import stage42_local_t100_easy_guard_repair as s42bi


def _row(speed: float, cv: float, alt: float) -> dict:
    return {
        "horizon": 100,
        "speed_causal": speed,
        "errors_eval_only": {
            "constant_velocity_causal_fd": cv,
            "damped_velocity_0p50": alt,
            "constant_position": alt,
            "damped_velocity_0p25": alt,
            "damped_velocity_0p75": alt,
            "constant_acceleration_causal": alt,
        },
    }


def test_thresholds_from_sources_include_high_quantiles() -> None:
    rows = [_row(float(i), 10.0, 5.0) for i in range(10)]
    thresholds = s42bi._thresholds_from_sources(rows, 100)
    assert thresholds[0] == 0.0
    assert thresholds[-1] == 9.0
    assert len(thresholds) >= 5


def test_source_robust_policy_requires_all_support_sources_safe() -> None:
    train = [_row(1.0, 10.0, 5.0)]
    support = {
        "safe": [_row(1.0, 10.0, 5.0), _row(2.0, 10.0, 5.0)],
        "harmful_easy": [_row(1.0, 1.0, 2.0), _row(2.0, 10.0, 5.0)],
    }
    selected = s42bi._select_source_robust_policy(train_windows=train, support_source_windows=support, horizon=100)
    assert selected["selected_policy"]["policy_name"] == "global_constant_velocity_causal_fd"
    assert selected["selected_policy"]["fallback_reason"] == "no_source_robust_policy_met_positive_gain_and_easy_guard_on_all_support_sources"


def test_gate_passes_for_repaired_payload() -> None:
    payload = {
        "source": "fresh_source_robust_easy_guard_repair",
        "bd_verdict": "stage42_bd_local_t100_source_inventory_pass",
        "bh_verdict": "stage42_bh_independent_t100_source_audit_partial",
        "repair_strategy": {"holdout_used_for_selection": False},
        "domain_summary": {"UCY": {}},
        "summary": {
            "ucy_t100_source_cv_supported": True,
            "ucy_t100_max_easy_degradation": 0.01,
            "ucy_t100_min_improvement_vs_fallback": 0.1,
            "blocked_domains": ["ETH_UCY", "TrajNet"],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "holdout_used_for_threshold": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "global_t100_positive_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bi._gate(payload)
    assert gate["passed"] == gate["total"]
