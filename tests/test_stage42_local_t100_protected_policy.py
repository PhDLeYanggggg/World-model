from src import stage42_local_t100_protected_policy as s42bg


def test_policy_metrics_fallback_has_zero_improvement() -> None:
    rows = [
        {
            "horizon": 100,
            "speed_causal": 1.0,
            "accel_causal": 0.0,
            "errors_eval_only": {
                "constant_velocity_causal_fd": 10.0,
                "constant_position": 5.0,
            },
        },
        {
            "horizon": 100,
            "speed_causal": 2.0,
            "accel_causal": 0.0,
            "errors_eval_only": {
                "constant_velocity_causal_fd": 20.0,
                "constant_position": 10.0,
            },
        },
    ]
    metrics = s42bg._policy_metrics(rows, s42bg._global_selector("constant_velocity_causal_fd"))
    assert metrics["improvement_vs_fallback"] == 0.0
    assert metrics["switch_rate"] == 0.0


def test_validation_selection_prefers_safe_improving_policy() -> None:
    train = [
        {
            "horizon": 100,
            "speed_causal": 1.0,
            "accel_causal": 0.0,
            "errors_eval_only": {
                "constant_velocity_causal_fd": 10.0,
                "constant_position": 4.0,
                "damped_velocity_0p25": 5.0,
                "damped_velocity_0p50": 6.0,
                "damped_velocity_0p75": 8.0,
                "constant_acceleration_causal": 40.0,
            },
        }
    ]
    val = [
        {
            "horizon": 100,
            "speed_causal": 1.0,
            "accel_causal": 0.0,
            "errors_eval_only": {
                "constant_velocity_causal_fd": 10.0,
                "constant_position": 4.0,
                "damped_velocity_0p25": 5.0,
                "damped_velocity_0p50": 6.0,
                "damped_velocity_0p75": 8.0,
                "constant_acceleration_causal": 40.0,
            },
        }
    ]
    selected = s42bg._select_policy_on_validation(train_windows=train, val_windows=val, horizon=100)
    assert selected["selected_policy"]["policy_name"] == "global_constant_position"


def test_validation_selection_falls_back_when_easy_harmful() -> None:
    train = [
        {
            "horizon": 100,
            "speed_causal": 1.0,
            "accel_causal": 0.0,
            "errors_eval_only": {
                "constant_velocity_causal_fd": 10.0,
                "constant_position": 9.0,
                "damped_velocity_0p25": 8.0,
                "damped_velocity_0p50": 7.0,
                "damped_velocity_0p75": 8.0,
                "constant_acceleration_causal": 40.0,
            },
        }
    ]
    val = [
        {
            "horizon": 100,
            "speed_causal": 1.0,
            "accel_causal": 0.0,
            "errors_eval_only": {
                "constant_velocity_causal_fd": 1.0,
                "constant_position": 2.0,
                "damped_velocity_0p25": 2.0,
                "damped_velocity_0p50": 2.0,
                "damped_velocity_0p75": 2.0,
                "constant_acceleration_causal": 2.0,
            },
        },
        {
            "horizon": 100,
            "speed_causal": 2.0,
            "accel_causal": 0.0,
            "errors_eval_only": {
                "constant_velocity_causal_fd": 100.0,
                "constant_position": 90.0,
                "damped_velocity_0p25": 80.0,
                "damped_velocity_0p50": 70.0,
                "damped_velocity_0p75": 80.0,
                "constant_acceleration_causal": 200.0,
            },
        },
    ]
    selected = s42bg._select_policy_on_validation(train_windows=train, val_windows=val, horizon=100)
    assert selected["selected_policy"]["policy_name"] == "global_constant_velocity_causal_fd"
    assert selected["selected_policy"]["fallback_reason"] == "no_validation_safe_policy_exceeded_fallback"


def test_gate_requires_global_t100_claim_blocked() -> None:
    payload = {
        "source": "fresh_source_cv_protected_policy",
        "be_verdict": "stage42_be_local_t100_conversion_readiness_pass",
        "bf_verdict": "stage42_bf_local_t100_schema_conversion_pass",
        "feature_store_manifest": {"materialized_large_feature_store_written": False},
        "summary": {
            "candidate_sources": 1,
            "policy_window_sources": 1,
            "t100_policy_windows": 10,
            "source_cv_domains_evaluated": ["UCY"],
            "source_cv_domains_blocked": ["ETH_UCY"],
            "ucy_t100_source_cv_supported": True,
            "training_run": True,
            "training_type": "validation_selected_baseline_family_policy",
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
            "t100_positive_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bg._gate(payload)
    assert gate["passed"] == gate["total"]
