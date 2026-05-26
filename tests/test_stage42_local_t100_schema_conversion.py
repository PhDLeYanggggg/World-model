from src import stage42_local_t100_schema_conversion as s42bf


def test_baseline_prediction_constant_velocity() -> None:
    prev2 = {"x": 0.0, "y": 0.0}
    prev = {"x": 1.0, "y": 1.0}
    cur = {"x": 3.0, "y": 2.0}
    px, py = s42bf._baseline_prediction("constant_velocity_causal_fd", prev2, prev, cur, 5)
    assert px == 13.0
    assert py == 7.0


def test_baseline_prediction_constant_acceleration() -> None:
    prev2 = {"x": 0.0, "y": 0.0}
    prev = {"x": 1.0, "y": 0.0}
    cur = {"x": 3.0, "y": 0.0}
    px, py = s42bf._baseline_prediction("constant_acceleration_causal", prev2, prev, cur, 2)
    assert px == 9.0
    assert py == 0.0


def test_source_cv_baseline_audit_selects_validation_best() -> None:
    metrics = {
        "a": {
            "by_horizon": {
                "100": {
                    "baselines": {
                        "constant_velocity_causal_fd": {"mean_fde": 10.0},
                        "constant_position": {"mean_fde": 5.0},
                    },
                    "windows": 3,
                }
            }
        },
        "b": {
            "by_horizon": {
                "100": {
                    "baselines": {
                        "constant_velocity_causal_fd": {"mean_fde": 12.0},
                        "constant_position": {"mean_fde": 6.0},
                    },
                    "windows": 4,
                }
            }
        },
    }
    plan = {"domains": {"UCY": {"folds": [{"validation_source": "a", "holdout_source": "b"}]}}}
    audit = s42bf._source_cv_baseline_audit(metrics, plan)
    fold = audit["domains"]["UCY"]["folds"][0]
    assert fold["selected_baseline_from_validation"] == "constant_position"
    assert fold["holdout_improvement_vs_constant_velocity"] == 0.5


def test_gate_passes_for_conversion_payload() -> None:
    payload = {
        "source": "fresh_in_memory_schema_conversion",
        "be_verdict": "stage42_be_local_t100_conversion_readiness_pass",
        "source_metrics": {
            "a": {"by_horizon": {"100": {"baselines": {"constant_velocity_causal_fd": {"mean_fde": 1.0}}}}}
        },
        "source_cv_audit": {"domains": {"UCY": {}}},
        "summary": {
            "candidate_sources": 1,
            "converted_sources": 1,
            "total_eval_windows_by_horizon": {"50": 10},
            "t100_eval_windows": 5,
            "source_cv_domains_evaluated": ["UCY"],
            "materialized_feature_store_written": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "t100_positive_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bf._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bf_local_t100_schema_conversion_pass"
