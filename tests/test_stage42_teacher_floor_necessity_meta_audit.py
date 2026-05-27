from src import stage42_teacher_floor_necessity_meta_audit as hb


def test_gate_passes_for_teacher_floor_meta_audit_payload() -> None:
    payload = {
        "source": "fresh_stage42_hb_teacher_floor_necessity_meta_audit",
        "input_status": {
            "a": {"gate_passed": True},
            "b": {"gate_passed": True},
        },
        "floor_taxonomy": {
            "protected_composite_deployable": True,
            "ungated_endpoint_unsafe": True,
            "ungated_full_waypoint_unsafe": True,
            "teacher_raw_not_deployable_due_proximity": True,
            "global_teacher_floor_required": True,
            "teacher_floor_context_required": True,
            "partial_t50_floor_relaxation_supported": True,
            "global_floor_removal_allowed": False,
            "floor_free_neural_deployable": False,
            "proximity_guard_required_for_safety_sensitive_full_waypoint": True,
            "full_waypoint_claim_guard_blocks_ungated": True,
            "full_waypoint_linter_clean": True,
        },
        "summary": {
            "global_floor_removal_allowed": False,
            "floor_free_neural_deployable": False,
            "m3w_neural_v1_protected_all": 0.2,
            "m3w_neural_v1_protected_t50": 0.1,
            "m3w_neural_v1_protected_hard": 0.2,
            "m3w_neural_v1_protected_easy": 0.0,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = hb._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_hb_teacher_floor_necessity_meta_audit_pass"


def test_current_teacher_floor_meta_audit_inputs_pass() -> None:
    payload = hb._build_payload()
    gate = payload["stage42_hb_gate"]
    assert gate["passed"] == gate["total"]
    assert payload["floor_taxonomy"]["global_teacher_floor_required"] is True
    assert payload["floor_taxonomy"]["partial_t50_floor_relaxation_supported"] is True
    assert payload["floor_taxonomy"]["global_floor_removal_allowed"] is False
    assert payload["floor_taxonomy"]["floor_free_neural_deployable"] is False
    assert payload["floor_taxonomy"]["full_waypoint_linter_clean"] is True


def test_pct_formats_none_and_float() -> None:
    assert hb._pct(None) == "n/a"
    assert hb._pct(0.1234) == "12.34%"
