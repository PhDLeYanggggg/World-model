from src import stage42_source_acquisition_status as bv


def test_user_actions_cover_all_active_blockers() -> None:
    blockers = [
        {"blocker_id": "a", "status": "blocked", "next_action": "do a"},
        {"blocker_id": "b", "status": "blocked_narrowed", "next_action": "do b"},
        {"blocker_id": "c", "status": "ready", "next_action": "do c"},
    ]
    actions = bv._user_actions(blockers)
    assert [row["blocker_id"] for row in actions] == ["a", "b"]
    assert all(row["allowed_next_command"].startswith("rerun") for row in actions)


def test_gate_requires_no_overclaim_and_user_actions() -> None:
    payload = {
        "source": "fresh_stage42_bv_source_acquisition_status",
        "artifact_inputs": {
            "br": {"exists": True, "verdict": "stage42_br_calibrated_t50_source_support_gap_audit_pass"},
            "bt": {"exists": True, "verdict": "stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed"},
            "bm": {"exists": True, "verdict": "stage42_bm_eth_person_terms_audit_pass_claim_blocked"},
            "bk": {"exists": True, "verdict": "stage42_bk_post_bj_local_source_verification_pass"},
            "bu": {"exists": True, "verdict": "stage42_bu_ucy_students_t50_source_support_pass_blocker_narrowed"},
            "bj": {"exists": True, "verdict": "stage42_bj_post_bi_t100_source_package_pass"},
            "bn": {"exists": True, "verdict": "stage42_bn_source_time_geometry_calibration_pass"},
        },
        "blocker_matrix": [{"status": "blocked"} for _ in range(5)],
        "user_action_required": [{"action": "x"} for _ in range(4)],
        "official_references": [{"id": str(i)} for i in range(4)],
        "summary": {
            "ucy_students_blocker_narrowed": True,
            "eth_seq_blocker_resolved": False,
            "trajnet_raw_long_source_resolved": False,
            "auto_download_executed": False,
            "registry_only_counted_as_converted": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = bv._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bv_source_acquisition_status_pass_blockers_actionable"
