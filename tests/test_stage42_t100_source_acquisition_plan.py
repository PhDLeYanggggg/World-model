from src import stage42_t100_source_acquisition_plan as s42bc


def test_download_policy_blocks_terms_and_restricted_license() -> None:
    candidate = {
        "requires_login_or_terms": True,
        "official_source_found": True,
        "auto_download_allowed": False,
        "license_access_summary": "CC BY-NC-ND 3.0 restricted",
    }
    policy = s42bc._download_policy(candidate)
    assert policy["auto_download_allowed"] is False
    assert policy["download_status"] == "not_run"
    assert any("terms" in reason for reason in policy["blocked_reasons"])
    assert any("restrict" in reason for reason in policy["blocked_reasons"])


def test_priority_score_rewards_unsupported_target_domain() -> None:
    candidate = {
        "official_source_found": True,
        "target_domains": ["TrajNet"],
        "t100_repair_value": "high",
        "metric_time_value": "low_without_source_specific_fps_stride",
        "requires_login_or_terms": False,
        "license_access_summary": "manual",
    }
    local = {"local_path_found": True}
    score = s42bc._priority_score(candidate, local, {"unsupported_t100_domains": ["TrajNet"]})
    assert score >= 80


def test_user_actions_include_domain_gap_and_candidate_actions() -> None:
    candidates = [
        {
            "source": "fresh",
            "id": "trajnetpp",
            "dataset_name": "TrajNet++",
            "target_domains": ["TrajNet"],
            "priority_group": "A",
            "official_url": "official",
            "local_status": {"found_paths": ["local"]},
            "download_policy": {
                "blocked_reasons": ["terms"],
                "safe_next_step": "verify",
            },
        }
    ]
    actions = s42bc._user_actions(candidates, {"additional_t100_sources_needed_by_domain": {"TrajNet": 1}})
    assert any(action["target"] == "TrajNet" for action in actions)
    assert any(action["target"] == "trajnetpp" for action in actions)


def test_gate_requires_no_download_and_no_overclaim() -> None:
    payload = {
        "source": "fresh_synthesis_from_stage42_bb_plus_official_web_pages",
        "bb_verdict": "stage42_bb_t100_data_gap_audit_pass_with_data_blocker",
        "web_sources_used": [{}, {}, {}],
        "user_action_required": [{"target": "A"}],
        "summary": {
            "candidate_sources": 6,
            "official_sources_found": 5,
            "local_path_found_sources": 4,
            "t100_repair_priority_order": ["a", "b", "c"],
            "user_action_required_count": 3,
            "auto_download_executed": False,
            "auto_download_allowed_sources": [],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "no_raw_download_executed": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bc._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bc_t100_source_acquisition_plan_pass"
