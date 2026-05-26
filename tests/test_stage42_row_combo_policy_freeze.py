from src import stage42_row_combo_policy_freeze as s


def test_stage42_s_policy_artifact_and_boundaries():
    result = s.run_stage42_row_combo_policy_freeze()
    assert result["policy_hash"]
    assert result["policy"]["selection_scope"] == "validation_domain_horizon_slice_only"
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False


def test_stage42_s_gate_and_stress_reported():
    result = s.run_stage42_row_combo_policy_freeze()
    gate = result["stage42_s_gate"]
    assert gate["gates"]["policy_hash_recorded"]
    assert gate["gates"]["validation_only_policy"]
    assert gate["gates"]["stress_complete"]
    assert "by_domain" in result["stress"]
    assert "by_horizon" in result["stress"]


def test_stage42_s_policy_is_lightweight_no_npz_payloads():
    result = s.run_stage42_row_combo_policy_freeze()
    text = s.POLICY_JSON.read_text(encoding="utf-8")
    assert "combo_test_ade" not in text
    assert "floor_test_ade" not in text
    assert result["cache_manifest"]
