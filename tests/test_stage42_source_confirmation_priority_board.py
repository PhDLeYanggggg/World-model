from src.stage42_source_confirmation_priority_board import _gate, _priority_score, _rank_rows


def test_priority_prefers_calibrated_t100_and_raw_path():
    row = {
        "dataset_id": "ucy_crowd_original",
        "domain": "UCY",
        "official_url": "https://example.org/ucy",
        "after_terms_potential": {"estimated_t50_windows": 100, "estimated_t100_windows": 50, "source_cv_after_terms": True},
        "calibrated_subset_after_terms": {
            "restricted_metric_time_candidates_after_terms": 1,
            "calibrated_t50_windows_after_terms": 100,
            "calibrated_t100_windows_after_terms": 50,
        },
        "suggested_local_paths_for_user_review": [
            {"exists": True, "is_raw_source_candidate": True, "is_derived_or_cache": False}
        ],
    }
    low = {
        "dataset_id": "other",
        "domain": "other",
        "official_url": "user_or_web_verified_official_url_required",
        "after_terms_potential": {"estimated_t50_windows": 0, "estimated_t100_windows": 0},
        "calibrated_subset_after_terms": {},
        "suggested_local_paths_for_user_review": [],
    }
    assert _priority_score(row, [], {"metric_time_subset_hint": True}) > _priority_score(low, [], {})


def test_rank_rows_preserves_blocked_contract_status():
    contract = {
        "contract_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "official_url": "https://example.org/ucy",
                "contract_status": "blocked_until_user_terms_path_source_confirmation",
                "contract_conversion_ready_now": False,
                "confirmation": {"missing_fields": ["terms_accepted_by_user", "local_path"]},
                "after_terms_potential": {"estimated_t50_windows": 100, "estimated_t100_windows": 50},
                "calibrated_subset_after_terms": {
                    "restricted_metric_time_candidates_after_terms": 1,
                    "calibrated_t50_windows_after_terms": 100,
                    "calibrated_t100_windows_after_terms": 50,
                },
                "suggested_local_paths_for_user_review": [],
            }
        ]
    }
    rows = _rank_rows(contract, {"plan_rows": []}, {"candidate_rows": []})
    assert rows[0]["priority_rank"] == 1
    assert rows[0]["contract_conversion_ready_now"] is False
    assert rows[0]["conversion_executed"] is False
    assert rows[0]["missing_user_fields"] == ["terms_accepted_by_user", "local_path"]


def test_gate_passes_no_conversion_priority_board():
    payload = {
        "source": "fresh_stage42_gn_source_confirmation_priority_board",
        "input_status": {
            "contract_exists": True,
            "harness_exists": True,
            "calibrated_plan_exists": True,
            "calibration_manifest_exists": True,
            "contract_row_count": 1,
        },
        "priority_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "missing_user_fields": ["terms_accepted_by_user"],
            }
        ],
        "summary": {
            "ready_now": 0,
            "blocked_now": 1,
            "total_t50_after_terms": 100,
            "total_t100_after_terms": 50,
            "calibrated_t50_after_terms": 100,
            "calibrated_t100_after_terms": 50,
            "top_priority_dataset": "ucy_crowd_original",
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "converted_dataset_claim_allowed": False,
            "restricted_metric_time_claim_allowed_now": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required_written": True,
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gn_source_confirmation_priority_board_pass"
