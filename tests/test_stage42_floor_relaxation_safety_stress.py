from src.stage42_floor_relaxation_safety_stress import run_stage42_floor_relaxation_safety_stress


def test_stage42_floor_relaxation_safety_stress_outputs():
    payload = run_stage42_floor_relaxation_safety_stress()
    assert payload["source"] == "fresh_stage42_gt_floor_relaxation_safety_stress"
    assert payload["stage42_gt_gate"]["passed"] == payload["stage42_gt_gate"]["total"]
    assert payload["summary"]["target_union_rows"] > 0
    assert payload["summary"]["global_floor_removal_allowed"] is False
    assert payload["claim_boundary"]["floor_free_neural_deployable"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
    assert "target_union_t50" in payload["stress_tests"]
