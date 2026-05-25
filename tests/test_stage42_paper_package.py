from src import stage42_paper_package as s42f


def test_stage42f_claim_matrix_keeps_negative_claims() -> None:
    evidence = {
        "data_calibration": {"global_metric_claim_allowed": False, "global_seconds_claim_allowed": False},
        "external_validation": {"all": 0.1, "t50": 0.1, "hard_failure": 0.1, "easy_degradation": 0.0, "ungated_easy_degradation": 1.0},
        "full_waypoint": {"ade_all": 0.1, "ade_t50": 0.1, "ade_t100_raw_frame_diagnostic": 0.1, "positive_domains": ["A", "B"]},
        "ablation": {"required_ablation_coverage_gate": True, "all_components_retrained_inside_stage42_d": False},
        "safety_floor": {"floor_necessity_conclusion": "teacher_floor_required_for_current_deployment"},
    }
    claims = s42f._claim_matrix(evidence)
    statuses = {row["status"] for row in claims}
    assert "rejected" in statuses
    assert "not_supported" in statuses


def test_stage42f_gate_not_full_a_journal_without_metric_and_retrain() -> None:
    data = {"summary": {"stage42_b_external_validation_ready": True, "global_metric_claim_allowed": False, "global_seconds_claim_allowed": False}}
    external = {"stage42_b_gate": {"verdict": "stage42_b_external_validation_pass_protected_neural_not_ungated"}}
    full = {"stage42_c_gate": {"verdict": "stage42_c_full_waypoint_dynamics_pass"}}
    ablation = {
        "stage42_d_gate": {"verdict": "stage42_d_causal_ablation_evidence_pass_with_retrain_boundary"},
        "full_retrain_boundary": {"all_components_retrained_inside_stage42_d": False},
    }
    safety = {"stage42_e_gate": {"verdict": "stage42_e_safety_floor_research_pass"}}
    evidence = {
        "full_waypoint": {"ade_all": 0.1, "ade_t50": 0.1},
        "safety_floor": {"best_all": 0.1, "best_easy_degradation": 0.0},
    }
    claims = [{"status": "rejected"}, {"status": "not_supported"}]
    gate = s42f._gate(data, external, full, ablation, safety, evidence, claims)
    assert gate["passed"] == gate["total"]
    assert gate["full_a_journal_ready"] is False


def test_stage42f_current_facts_block_overclaim() -> None:
    text = "\n".join(s42f.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text
