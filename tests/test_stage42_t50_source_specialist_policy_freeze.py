from src import stage42_t50_source_specialist_policy_freeze as im


def _ik_payload():
    return {
        "summary": {
            "rows": 10,
            "ade_all": 0.1,
            "ade_t50": 0.2,
            "ade_t50_ci_low": 0.1,
            "ade_t100_raw_frame_diagnostic": 0.3,
            "ade_hard_failure": 0.4,
            "ade_easy_degradation": 0.0,
            "fde_t50": 0.5,
            "fde_t50_ci_low": 0.4,
            "switch_rate": 0.2,
        },
        "source_rows": [{"source_file": "a", "t50_improvement": 0.1}],
        "by_domain": {"UCY": {"t50_improvement": 0.1}},
        "alignment": {"ucy_mask_matches_domain": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "source_specialist_claim_only": True,
            "independent_new_domain_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _il_payload():
    return {
        "summary": {
            "ucy_delta": {"after_t50": 0.1},
            "non_ucy_max_abs_delta": 0.0,
        },
        "supported_claims": [{"claim": "x"}],
        "blocked_claims": [{"claim": "y"}],
    }


def test_policy_payload_freezes_source_routing():
    policy = im._policy_payload(_ik_payload(), _il_payload())
    assert policy["selection_scope"] == "prevalidated_source_routing_no_new_threshold_selection"
    assert policy["routing_rule"]["default_route"] == "stage42ii_t50_gain_harm_ensemble"
    assert policy["routing_rule"]["ucy_route"] == "stage42x_row_aligned_ucy_full_waypoint_specialist"
    assert policy["routing_rule"]["uses_test_metrics_for_routing"] is False


def test_replay_matches_policy_to_ik_and_il():
    ik = _ik_payload()
    il = _il_payload()
    policy = im._policy_payload(ik, il)
    replay = im._replay(policy, ik, il)
    assert replay["metric_summary_exact_replay"] is True
    assert replay["source_rows_exact_replay"] is True
    assert replay["domain_rows_exact_replay"] is True
    assert replay["il_delta_audit_exact_replay"] is True


def test_gate_passes_for_frozen_source_specialist_policy():
    ik = _ik_payload()
    il = _il_payload()
    policy = im._policy_payload(ik, il)
    payload = {
        "inputs": {
            "stage42ik_gate": {"passed": 16, "total": 16},
            "stage42il_gate": {"passed": 16, "total": 16},
        },
        "frozen_policy": policy,
        "policy_artifact": {"sha256": "a" * 64},
        "policy_hash": "b" * 64,
        "replay": im._replay(policy, ik, il),
    }
    gate = im._gate(payload)
    assert gate["verdict"] == "stage42_im_t50_source_specialist_policy_freeze_pass"
    assert gate["passed"] == gate["total"]


def test_stage42_im_run_records_frozen_policy_boundary():
    result = im.run_stage42_t50_source_specialist_policy_freeze()
    gate = result["stage42_im_gate"]
    assert gate["gates"]["routing_rule_frozen"]
    assert gate["gates"]["compact_metric_replay_exact"]
    assert result["frozen_policy"]["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["frozen_policy"]["claim_boundary"]["stage5c_executed"] is False
    assert result["frozen_policy"]["claim_boundary"]["smc_enabled"] is False
