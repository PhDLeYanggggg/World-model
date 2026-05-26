from src import stage42_calibrated_t50_source_support_gap_audit as br


def test_family_summary_marks_leave_one_out_support_gap() -> None:
    bq_payload = {
        "fold_results": [
            {"fold": {"holdout_source": "ETH_seq_eth"}, "protected_ade": {"t50_improvement": 0.0, "all_improvement": 0.1}},
            {"fold": {"holdout_source": "ETH_seq_hotel"}, "protected_ade": {"t50_improvement": 0.0, "all_improvement": 0.1}},
            {"fold": {"holdout_source": "UCY_zara01"}, "protected_ade": {"t50_improvement": 0.0, "all_improvement": 0.1}},
            {"fold": {"holdout_source": "UCY_zara02"}, "protected_ade": {"t50_improvement": 0.0, "all_improvement": 0.0}},
            {"fold": {"holdout_source": "UCY_zara03"}, "protected_ade": {"t50_improvement": 0.0, "all_improvement": 0.0}},
            {"fold": {"holdout_source": "UCY_students03"}, "protected_ade": {"t50_improvement": 0.0, "all_improvement": 0.0}},
        ]
    }
    summary = br._family_summary(
        ["ETH_seq_eth", "ETH_seq_hotel", "UCY_zara01", "UCY_zara02", "UCY_zara03", "UCY_students03"],
        bq_payload,
    )
    assert summary["ETH_seq"]["additional_calibrated_sources_needed"] == 1
    assert summary["UCY_zara"]["additional_calibrated_sources_needed"] == 0
    assert summary["UCY_zara"]["primary_blocker"] == "enough_family_sources_but_no_safe_positive_t50_policy"
    assert summary["UCY_students"]["additional_calibrated_sources_needed"] == 2


def test_local_candidate_summary_preserves_terms_blocker() -> None:
    bm = {"summary": {"official_terms_verified": False, "license_terms_confirmed": False, "next_stage_official_conversion_allowed": False}}
    bk = {
        "summary": {
            "eth_person_xml_candidates": ["ETH-Person/data/a.xml"],
            "can_repair_eth_ucy_with_local_candidates_after_license_confirmation": True,
            "trajnet_t100_capable_files": 0,
            "trajnet_independent_t100_groups": 0,
        }
    }
    summary = br._local_candidate_summary(bm, bk)
    assert summary["eth_person_candidate_count"] == 1
    assert summary["eth_person_terms_verified"] is False
    assert summary["eth_person_official_conversion_allowed"] is False
    assert summary["trajnet_local_long_track_blocker"] is True


def test_gate_passes_only_as_gap_audit_not_positive_claim() -> None:
    payload = {
        "source": "fresh_calibrated_t50_source_support_gap_audit",
        "bq_verdict": "stage42_bq_calibrated_subset_t50_support_repair_pass_t50_nonharm_limited_positive",
        "summary": {
            "families_audited": 3,
            "unsupported_family_holdout_count": 2,
            "bq_t50_min": 0.0,
            "bq_easy_max": 0.0,
            "bq_positive_t50_fold_count": 0,
            "eth_person_candidate_count": 5,
            "trajnet_t100_capable_files": 0,
            "eth_person_terms_verified": False,
        },
        "user_action_required": [{"family": "ETH_seq"}],
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = br._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_br_calibrated_t50_source_support_gap_audit_pass"
