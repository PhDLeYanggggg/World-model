from src import stage42_floor_relaxability_audit as bx


def test_positive_and_easy_safe_requires_easy_limit() -> None:
    ok = {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.0, "easy_degradation": 0.0}
    bad_easy = dict(ok, easy_degradation=0.03)
    bad_gain = dict(ok, all_improvement=-0.01)
    assert bx._is_positive_and_easy_safe(ok, horizon=50)
    assert not bx._is_positive_and_easy_safe(bad_easy, horizon=50)
    assert not bx._is_positive_and_easy_safe(bad_gain, horizon=50)


def test_slice_decision_blocks_missing_validation_support() -> None:
    row = {
        "val_rows": 0,
        "test_rows": 100,
        "val_metric": {},
        "test_metric": {"all_improvement": 0.2, "t50_improvement": 0.2, "hard_failure_improvement": 0.2, "easy_degradation": 0.0},
    }
    decision = bx._slice_decision("UCY|50", row)
    assert decision["status"] == "blocked_no_validation_support"
    assert decision["deployment_decision"] == "fallback_required"


def test_slice_decision_allows_validation_and_test_safe_slice() -> None:
    metric = {"all_improvement": 0.2, "t50_improvement": 0.2, "hard_failure_improvement": 0.2, "easy_degradation": 0.0}
    row = {"val_rows": 100, "test_rows": 100, "val_metric": metric, "test_metric": metric}
    decision = bx._slice_decision("TrajNet|50", row)
    assert decision["status"] == "relaxable_under_validation_rule"
    assert decision["deployment_decision"] == "fallback_relaxable_for_this_slice"


def test_gate_passes_with_limited_relaxable_and_blocked_slices() -> None:
    payload = {
        "source": "fresh_stage42_bx_floor_relaxability_audit",
        "summary": {
            "slice_count": 3,
            "relaxable_slice_count": 1,
            "blocked_slice_count": 2,
            "blocked_no_validation_support": ["UCY|50"],
            "blocked_by_validation_safety": ["TrajNet|100"],
            "t50_relaxable_slices": ["TrajNet|50"],
            "t50_blocked_slices": [],
            "t100_relaxable_slices": [],
            "teacher_floor_context_required": True,
            "floor_free_neural_deployable": False,
            "source_blockers_active": 1,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = bx._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bx_floor_relaxability_audit_pass"


def test_gate_passes_when_t50_is_explicitly_blocked() -> None:
    payload = {
        "source": "fresh_stage42_bx_floor_relaxability_audit",
        "summary": {
            "slice_count": 3,
            "relaxable_slice_count": 1,
            "blocked_slice_count": 2,
            "blocked_no_validation_support": ["UCY|50"],
            "blocked_by_validation_safety": ["TrajNet|50"],
            "t50_relaxable_slices": [],
            "t50_blocked_slices": ["UCY|50", "TrajNet|50"],
            "t100_relaxable_slices": [],
            "teacher_floor_context_required": True,
            "floor_free_neural_deployable": False,
            "source_blockers_active": 1,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = bx._gate(payload)
    assert gate["passed"] == gate["total"]
