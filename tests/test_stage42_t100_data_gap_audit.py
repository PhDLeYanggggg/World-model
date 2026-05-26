from src import stage42_t100_data_gap_audit as s42bb


def test_t100_source_gap_counts_missing_safe_folds() -> None:
    support = {
        "status": "fresh_run",
        "fold_count": 4,
        "safe_positive_fold_count": 0,
        "max_easy_degradation": 0.39,
        "min_t100_improvement": 0.0,
        "supported_for_t100": False,
    }
    gap = s42bb._t100_source_gap("ETH_UCY", support, {"t100_groups": [{"group": "a"}, {"group": "b"}, {"group": "c"}]})
    assert gap["missing_safe_positive_folds"] == 2
    assert gap["additional_t100_capable_train_sources_needed"] == 2
    assert gap["blocker_type"] == "t100_easy_safety_not_stable_across_source_cv"


def test_t100_source_gap_insufficient_sources_for_not_run_domain() -> None:
    support = {
        "status": "not_run",
        "reason": "fewer_than_three_t100_capable_original_train_sources",
        "fold_count": 0,
        "supported_for_t100": False,
    }
    gap = s42bb._t100_source_gap("UCY", support, {"t100_groups": [{"group": "a"}, {"group": "b"}]})
    assert gap["additional_t100_capable_train_sources_needed"] == 1
    assert gap["blocker_type"] == "insufficient_t100_capable_original_train_sources"


def test_dataset_action_keeps_tgsim_diagnostic_only() -> None:
    action = s42bb._dataset_action(
        {
            "dataset_id": "tgsim",
            "dataset_name": "TGSIM",
            "raw_path_found": True,
            "converted_path_found": True,
            "calibration_state": "traffic_metric_diagnostic_only",
            "metric_claim_allowed": True,
            "seconds_claim_allowed": False,
            "official_hint": "official",
            "legal_status": {},
        },
        {},
    )
    assert action["next_action"] == "user_action_or_source_specific_repair_required"
    assert any("Traffic metric diagnostic only" in reason for reason in action["reasons"])


def test_gate_passes_with_t100_blocker_and_safe_other_metrics() -> None:
    payload = {
        "source": "fresh_synthesis_from_stage42_ba_and_calibration",
        "ba_verdict": "stage42_ba_t100_source_cv_repair_pass_with_t100_blocker",
        "source_gaps": {"A": {"supported_for_t100": False}},
        "dataset_actions": [{"dataset_id": "a"}],
        "user_action_required": [{"target": "A"}],
        "summary": {
            "unsupported_t100_domains": ["A"],
            "any_t100_domain_supported": False,
            "final_all_positive_after_guard": True,
            "final_t50_positive_after_guard": True,
            "final_hard_positive_after_guard": True,
            "final_easy_safe_after_guard": True,
            "final_t100_positive_after_guard": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bb._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bb_t100_data_gap_audit_pass_with_data_blocker"
