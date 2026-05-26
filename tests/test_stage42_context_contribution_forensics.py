from src import stage42_context_contribution_forensics as ci


def test_stage42_ci_identifies_supported_and_mixed_contexts():
    result = ci.run_stage42_context_contribution_forensics()
    rows = {row["module"]: row for row in result["context_rows"]}
    assert rows["baseline_family_rollout_context"]["status"] == "supported_dominant_mechanism"
    assert rows["history_tokens"]["status"] == "supported_core_component"
    assert rows["stage37_teacher_floor_and_safe_switch"]["status"] == "supported_necessary_safety_mechanism"
    assert rows["goal_scene_context"]["status"] == "mixed_partial_not_main_claim"
    assert rows["neighbor_interaction_context"]["status"] == "mixed_weak_not_main_claim"
    assert not rows["goal_scene_context"]["main_claim_allowed"]
    assert not rows["neighbor_interaction_context"]["main_claim_allowed"]


def test_stage42_ci_gate_passes_without_overclaiming():
    result = ci.run_stage42_context_contribution_forensics()
    gate = result["stage42_ci_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["gates"]["dominant_mechanism_identified"]
    assert gate["gates"]["mixed_goal_scene_not_overclaimed"]
    assert gate["gates"]["mixed_neighbor_not_overclaimed"]
    assert gate["gates"]["jepa_not_overclaimed"]
    assert gate["gates"]["transformer_not_overclaimed"]
    assert gate["gates"]["stage5c_false"]
    assert gate["gates"]["smc_false"]


def test_stage42_ci_current_facts_keep_claim_boundary():
    facts = "\n".join(ci.CURRENT_FACTS)
    assert "不是 true 3D" in facts
    assert "raw-frame" in facts
    assert "Stage5C" in facts
    assert "SMC" in facts
