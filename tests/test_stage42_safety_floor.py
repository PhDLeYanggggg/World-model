from src import stage42_safety_floor as s42e


def test_stage42e_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42e.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42e_deployable_rejects_easy_harm() -> None:
    metrics = {
        "all_improvement": 0.3,
        "t50_improvement": 0.2,
        "hard_failure_improvement": 0.2,
        "easy_degradation": 0.5,
        "collision_delta_vs_floor_005": 0.0,
        "switch_rate": 0.5,
    }
    assert not s42e._deployable(metrics)


def test_stage42e_gate_requires_validation_only() -> None:
    result = {
        "cached_verified_context": {
            "stage42_b_verdict": "stage42_b_external_validation_pass_protected_neural_not_ungated",
            "stage42_c_verdict": "stage42_c_full_waypoint_dynamics_pass",
            "stage42_d_verdict": "stage42_d_causal_ablation_evidence_pass_with_retrain_boundary",
        },
        "switch_gate_rows": [{} for _ in range(6)],
        "bounded_residual_rows": [{} for _ in range(4)],
        "best_deployable_policy": {"test_deployable": True, "test_metrics": {"easy_degradation": 0.0}},
        "floor_necessity_analysis": {
            "conclusion": "teacher_floor_required_for_current_deployment",
            "ungated_endpoint_metrics_from_stage42_b": {"easy_degradation": 1.0},
            "ungated_full_waypoint_metrics_from_stage42_c": {"easy_degradation": 1.0},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "thresholds_selected_on_val": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42e._gate(result)
    assert gate["passed"] == gate["total"]
