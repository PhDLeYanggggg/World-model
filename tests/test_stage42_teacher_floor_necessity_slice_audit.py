from src import stage42_teacher_floor_necessity_slice_audit as jw


def test_teacher_floor_necessity_slice_audit_passes() -> None:
    payload = jw.run_stage42_teacher_floor_necessity_slice_audit(refresh_readmes=False)
    gate = payload["stage42_jw_gate"]
    assert gate["verdict"] == "stage42_jw_teacher_floor_necessity_slice_audit_pass"
    assert gate["passed"] == gate["total"]
    assert payload["summary"]["fallback_rows"] > 0
    assert payload["summary"]["fallback_exact_floor_rate"] >= 0.999


def test_teacher_floor_global_removal_remains_forbidden() -> None:
    payload = jw.run_stage42_teacher_floor_necessity_slice_audit(refresh_readmes=False)
    assert payload["summary"]["global_floor_removal_allowed"] is False
    assert payload["summary"]["floor_free_neural_deployable"] is False
    assert payload["claim_boundary"]["global_teacher_floor_removal_allowed"] is False
    assert payload["claim_boundary"]["floor_free_neural_deployable"] is False


def test_hard_switch_rate_is_at_least_easy_switch_rate() -> None:
    payload = jw.run_stage42_teacher_floor_necessity_slice_audit(refresh_readmes=False)
    assert payload["summary"]["hard_failure_switch_rate"] >= payload["summary"]["easy_switch_rate"]


def test_partial_t50_relaxation_is_limited_not_global() -> None:
    payload = jw.run_stage42_teacher_floor_necessity_slice_audit(refresh_readmes=False)
    relax = payload["summary"]["partial_t50_floor_relaxation"]
    assert relax["target_union_safety_pass"] is True
    assert relax["target_union_t50_improvement"] > 0
    assert payload["claim_boundary"]["partial_floor_relaxation_limited_to_guarded_t50"] is True
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
