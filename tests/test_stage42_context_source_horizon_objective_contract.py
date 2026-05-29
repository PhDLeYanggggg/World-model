from src import stage42_context_source_horizon_objective_contract as ka


def test_stage42_ka_gate_passes() -> None:
    payload = ka.run_stage42_context_source_horizon_objective_contract(refresh_readmes=False)
    gate = payload["stage42_ka_gate"]
    assert gate["verdict"] == "stage42_ka_context_source_horizon_objective_contract_pass"
    assert gate["passed"] == gate["total"]


def test_stage42_ka_blocks_global_context_promotion() -> None:
    payload = ka.run_stage42_context_source_horizon_objective_contract(refresh_readmes=False)
    summary = payload["summary"]
    assert summary["global_material_context_variants"] == []
    assert summary["claim_decision"] == "keep_scene_goal_neighbor_interaction_blocked_as_independent_main_claims"
    assert payload["claim_boundary"]["independent_context_main_claim_allowed"] is False


def test_stage42_ka_preserves_narrow_context_objectives() -> None:
    payload = ka.run_stage42_context_source_horizon_objective_contract(refresh_readmes=False)
    slices = payload["summary"]["narrow_auxiliary_context_slices"]
    assert {"variant": "history_only", "horizon": 10} in slices
    assert {"variant": "motion_goal_context", "horizon": 10} in slices
    matrix = payload["summary"]["horizon_objective_matrix"]
    assert matrix["10"]["objective_decision"] == "auxiliary_retrain_candidate_only"
    assert matrix["25"]["objective_decision"] == "diagnostic_router_only_not_baseline_family_positive"
    assert payload["summary"]["diagnostic_router_conflicts"] == [
        {
            "horizon": 25,
            "candidate": "baseline_plus_history_goal_neighbor",
            "decision": "diagnostic_router_only_not_baseline_family_positive",
        }
    ]


def test_stage42_ka_keeps_t50_t100_blocked_until_new_objective() -> None:
    payload = ka.run_stage42_context_source_horizon_objective_contract(refresh_readmes=False)
    matrix = payload["summary"]["horizon_objective_matrix"]
    assert matrix["50"]["objective_decision"] == "blocked_until_new_row_level_objective"
    assert matrix["100"]["objective_decision"] == "diagnostic_blocked_until_new_source_slice_objective"
    assert payload["summary"]["t50_oracle_headroom"] > 0
    assert payload["summary"]["t100_oracle_headroom"] > 0


def test_stage42_ka_claim_boundaries() -> None:
    payload = ka.run_stage42_context_source_horizon_objective_contract(refresh_readmes=False)
    claim = payload["claim_boundary"]
    assert claim["true_3d"] is False
    assert claim["foundation_world_model"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False
