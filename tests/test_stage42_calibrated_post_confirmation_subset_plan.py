from src import stage42_calibrated_post_confirmation_subset_plan as gh


def test_plan_rows_join_gf_and_bn_without_marking_ready_now():
    gf = {
        "source_plan_rows": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "source_id": "UCY_students03",
                "trajectory_file": "x",
                "t50_windows_after_terms": 10,
                "t100_windows_after_terms": 5,
                "technical_conversion_ready_after_terms": True,
                "source_cv_feasible_after_terms_for_domain": True,
                "causal_velocity_possible": True,
                "central_velocity_used": False,
                "legal_ready_now": False,
                "source_ready_now": False,
                "missing_user_fields": ["terms_accepted_by_user"],
            }
        ]
    }
    bn = {
        "source_records": [
            {
                "source_id": "UCY_students03",
                "source_specific_metric_time_evidence": True,
                "allowed_local_claim": "source_specific_annotation_step_meter_coordinate_evidence",
                "timing": {"annotation_fps": 2.5, "annotation_timestep_seconds": 0.4, "h50_annotation_seconds": 20.0},
                "homography": {"parseable": True},
                "coordinate": {"meter_coordinates_evidence": True},
            }
        ]
    }

    rows = gh._plan_rows(gf, bn)

    assert len(rows) == 1
    assert rows[0]["source_specific_metric_time_evidence"] is True
    assert rows[0]["restricted_metric_time_candidate_after_terms"] is True
    assert rows[0]["restricted_metric_time_ready_now"] is False
    assert rows[0]["conversion_executed"] is False
    assert rows[0]["horizon_seconds_after_legal_conversion"]["50"] == 20.0


def test_summary_counts_only_calibrated_windows():
    rows = [
        {
            "source_specific_metric_time_evidence": True,
            "restricted_metric_time_candidate_after_terms": True,
            "t50_windows_after_terms": 10,
            "t100_windows_after_terms": 3,
            "restricted_metric_time_ready_now": False,
            "source_ready_now": False,
        },
        {
            "source_specific_metric_time_evidence": True,
            "restricted_metric_time_candidate_after_terms": False,
            "t50_windows_after_terms": 50,
            "t100_windows_after_terms": 50,
            "restricted_metric_time_ready_now": False,
            "source_ready_now": False,
        },
    ]

    summary = gh._summary(rows)

    assert summary["planned_source_rows"] == 2
    assert summary["restricted_metric_time_candidates_after_terms"] == 1
    assert summary["calibrated_t50_windows_after_terms"] == 10
    assert summary["calibrated_t100_windows_after_terms"] == 3
    assert summary["restricted_ready_now"] == 0


def test_gate_passes_but_blocks_global_and_current_restricted_claims():
    payload = {
        "source": "fresh_stage42_gh_calibrated_post_confirmation_subset_plan",
        "input_status": {"gf_exists": True, "bn_exists": True},
        "summary": {
            "planned_source_rows": 2,
            "restricted_metric_time_candidates_after_terms": 1,
            "calibrated_t50_windows_after_terms": 10,
            "calibrated_t100_windows_after_terms": 5,
            "restricted_ready_now": 0,
            "source_ready_now": 0,
            "downloaded_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
        },
        "claim_boundary": gh.CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }

    gate = gh._gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["gates"]["global_claim_blocked"] is True
    assert gate["gates"]["restricted_claim_not_allowed_now"] is True
