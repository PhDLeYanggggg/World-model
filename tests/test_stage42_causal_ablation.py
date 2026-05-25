from src import stage42_causal_ablation as s42d


def test_stage42d_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42d.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42d_candidate_status_marks_unsafe() -> None:
    row = {
        "all_improvement": 0.4,
        "t50_improvement": 0.2,
        "hard_failure_improvement": 0.2,
        "easy_degradation": 1.2,
    }
    assert s42d._candidate_status(row) == "negative_unsafe"


def test_stage42d_gate_requires_source_labels() -> None:
    result = {
        "summary": {
            "stage42_b_verdict": "stage42_b_external_validation_pass_protected_neural_not_ungated",
            "stage42_c_verdict": "stage42_c_full_waypoint_dynamics_pass",
            "required_ablation_coverage_gate": True,
            "same_protocol_architecture_ablation_gate": True,
        },
        "fresh_ablation_rows": [
            {"source": "fresh_run", "ablation": f"row_{i}", "status": "positive_safe"}
            for i in range(5)
        ]
        + [{"source": "fresh_run", "ablation": "no_safe_floor_use_ungated_endpoint_neural", "status": "negative_unsafe"}],
        "cached_verified_required_ablation_rows": [{"source": "cached_verified"}],
        "cached_verified_architecture_rows": [{"source": "cached_verified"}],
        "full_retrain_boundary": {"all_components_retrained_inside_stage42_d": False},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42d._gate(result)
    assert gate["passed"] == gate["total"]
