from src import stage42_floor_free_proximity_guard_repair as hd


_PAYLOAD = None


def _payload():
    global _PAYLOAD
    if _PAYLOAD is None:
        _PAYLOAD = hd._build_payload()
    return _PAYLOAD


def test_stage42_hd_gate_passes() -> None:
    payload = _payload()
    gate = payload["stage42_hd_gate"]
    assert gate["passed"] == gate["total"]
    assert payload["summary"]["pre_guard_deployable_count"] == 0
    assert payload["summary"]["candidate_count"] >= 4
    assert payload["summary"]["causal_floor_fallback_used"] is True
    assert payload["summary"]["global_floor_removal_allowed"] is False


def test_stage42_hd_repair_uses_no_teacher_gate() -> None:
    payload = _payload()
    assert payload["summary"]["teacher_gate_used"] is False
    assert payload["validation_protocol"]["teacher_gate_used"] is False
    assert payload["claim_boundary"]["causal_floor_safety_fallback_still_required"] is True


def test_stage42_hd_best_row_has_guard_metadata() -> None:
    payload = _payload()
    best_family = payload["summary"]["best_post_guard_family"]
    rows = [row for row in payload["repair_rows"] if row["family"] == best_family]
    assert rows
    metrics = rows[0]["test_metrics"]
    assert "guarded_off_rate" in metrics
    assert "raw_switch_rate" in metrics
    assert "collision_delta_vs_floor_005" in metrics
