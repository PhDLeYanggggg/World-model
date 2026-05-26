from src import stage42_unified_ablation_evidence as y


def test_delta_treats_easy_degradation_as_ablation_minus_full():
    full = {"ade_all": 0.5, "ade_t50": 0.4, "ade_hard_failure": 0.3, "ade_easy_degradation": 0.01}
    ablated = {"ade_all": 0.2, "ade_t50": 0.1, "ade_hard_failure": 0.0, "ade_easy_degradation": 0.03}
    delta = y._delta(full, ablated)
    assert delta["ade_all"] == 0.3
    assert delta["ade_t50"] == 0.30000000000000004
    assert delta["ade_hard_failure"] == 0.3
    assert delta["ade_easy_degradation"] == 0.019999999999999997


def test_row_marks_positive_component_contribution():
    full = {"ade_all": 0.5, "ade_t50": 0.4, "ade_hard_failure": 0.3, "ade_easy_degradation": 0.0}
    ablated = {"ade_all": 0.1, "ade_t50": 0.4, "ade_hard_failure": 0.3, "ade_easy_degradation": 0.0}
    row = y._row("ablate", "unit", ablated, full, "test")
    assert row["positive_component_contribution"] is True
    assert row["loss_vs_stage42x_full"]["ade_all"] == 0.4


def test_stage42_y_run_records_safety_and_no_overclaim():
    result = y.run_stage42_unified_ablation_evidence()
    gate = result["stage42_y_gate"]
    assert gate["gates"]["stage42x_prereq_pass"]
    assert gate["gates"]["ucy_source_contribution_positive"]
    assert gate["gates"]["safety_floor_necessity_diagnosed"]
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
