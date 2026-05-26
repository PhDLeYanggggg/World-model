from src import stage42_ucy_students_t50_source_support as bu


def test_canonical_independent_sources_deduplicates_alternates() -> None:
    rows = [
        {"source_id": "a_short", "independent_key": "A", "path_exists": True, "t50_capable": True, "horizon_counts": {"50": 5}, "rows": 10},
        {"source_id": "a_long", "independent_key": "A", "path_exists": True, "t50_capable": True, "horizon_counts": {"50": 50}, "rows": 100},
        {"source_id": "b_none", "independent_key": "B", "path_exists": True, "t50_capable": False, "horizon_counts": {"50": 0}, "rows": 100},
    ]
    canonical = bu._canonical_independent_sources(rows)
    assert sorted(canonical) == ["A"]
    assert canonical["A"]["source_id"] == "a_long"


def test_gate_passes_blocker_narrowed_payload() -> None:
    payload = {
        "source": "fresh_ucy_students_t50_source_support",
        "br_verdict": "stage42_br_calibrated_t50_source_support_gap_audit_pass",
        "bs_verdict": "stage42_bs_ucy_zara_t50_family_policy_pass_positive",
        "bt_verdict": "stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed",
        "candidate_audit": [
            {"t50_capable": True},
            {"t50_capable": True},
            {"t50_capable": True},
        ],
        "summary": {
            "local_candidates_audited": 5,
            "independent_t50_capable_source_count": 2,
            "source_cv_ready": False,
            "ucy_students_t50_support_repaired": False,
            "additional_independent_t50_sources_still_needed": 1,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": [{"action": "provide_source"}],
    }
    gate = bu._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bu_ucy_students_t50_source_support_pass_blocker_narrowed"
