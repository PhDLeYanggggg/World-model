from src import stage42_t50_static_expert_combo as q


def test_stage42_q_gate_and_claim_boundaries():
    result = q.run_stage42_t50_static_expert_combo()
    gate = result["stage42_q_gate"]
    assert gate["gates"]["cached_reports_available"]
    assert gate["gates"]["complementarity_detected"]
    assert gate["gates"]["row_level_combo_not_claimed_complete"]
    assert gate["gates"]["stage5c_false"]
    assert gate["gates"]["smc_false"]
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False


def test_stage42_q_reports_preflight_and_row_cache_need():
    result = q.run_stage42_t50_static_expert_combo()
    assert result["row_level_combo_status"]["source"] == "attempted_not_completed"
    assert "row_prediction_cache" in result["row_level_combo_status"]["next_action"]
    assert result["diagnostic_envelope"]["source"] == "diagnostic_only_not_deployable"


def test_stage42_q_detects_j_p_complementarity():
    result = q.run_stage42_t50_static_expert_combo()
    c = result["complementarity"]
    assert c["p_beats_j_all"]
    assert c["p_beats_j_hard"]
    assert c["j_beats_p_t50"]
    assert c["p_t50_seed_ci_low_negative"]
