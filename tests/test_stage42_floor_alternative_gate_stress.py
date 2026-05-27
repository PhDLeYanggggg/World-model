from src import stage42_floor_alternative_gate_stress as hc


def test_failure_reasons_detect_easy_and_collision() -> None:
    metrics = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.03,
        "collision_delta_vs_floor_005": 0.02,
        "switch_rate": 0.5,
    }
    reasons = hc._failure_reasons(metrics)
    assert "easy_degradation_over_2pct" in reasons
    assert "near_collision_delta_over_1pp" in reasons
    assert hc._is_deployable(metrics) is False


def test_gate_passes_for_current_floor_alternative_stress() -> None:
    payload = hc._build_payload()
    gate = payload["stage42_hc_gate"]
    assert gate["passed"] == gate["total"]
    assert payload["summary"]["floor_free_deployable_count"] == 0
    assert payload["summary"]["teacher_dependent_deployable_count"] >= 1
    assert payload["summary"]["global_floor_removal_allowed"] is False
    assert payload["summary"]["floor_free_neural_deployable"] is False


def test_rows_are_grouped_by_floor_free_and_teacher_dependent() -> None:
    rows = hc._rows_by_family({"switch_gate_rows": [], "bounded_residual_rows": []})
    assert rows == []
    payload = hc._build_payload()
    types = {row["family_type"] for row in payload["candidate_rows"]}
    assert "floor_free_switch_gate" in types
    assert "teacher_dependent_switch_gate" in types
    assert "floor_free_bounded_residual" in types
