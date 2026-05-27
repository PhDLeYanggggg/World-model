from src.stage42_post_confirmation_conversion_plan import (
    _gate,
    _missing_user_fields,
    _source_plan_rows,
    _source_score,
)


def test_missing_user_fields_requires_explicit_terms_and_source_identity():
    row = {
        "user_confirmation": {
            "terms_accepted_by_user": False,
            "terms_acceptance_date": "",
            "allowed_use": "",
            "local_path": "",
            "source_identity": "",
            "redistribution_allowed": "unknown",
            "derived_data_allowed": "unknown",
            "confirmed_by_user": "",
        }
    }

    missing = _missing_user_fields(row)

    assert "terms_accepted_by_user" in missing
    assert "local_path" in missing
    assert "source_identity" in missing
    assert "redistribution_allowed" in missing
    assert "derived_data_allowed" in missing


def test_source_score_prefers_h100_source_cv_and_causal_velocity():
    low = {
        "horizon_counts": {"50": 20, "100": 0},
        "technical_conversion_ready_after_terms": True,
        "causal_velocity_possible": True,
        "central_velocity_used": False,
    }
    high = {
        "horizon_counts": {"50": 20, "100": 30},
        "technical_conversion_ready_after_terms": True,
        "causal_velocity_possible": True,
        "central_velocity_used": False,
    }

    assert _source_score(high, True) > _source_score(low, False)


def test_source_plan_rows_do_not_mark_prefill_as_ready():
    intake = {
        "datasets": [
            {
                "dataset_id": "ucy_crowd_original",
                "domain": "UCY",
                "user_confirmation": {
                    "terms_accepted_by_user": False,
                    "terms_acceptance_date": "",
                    "allowed_use": "",
                    "local_path": "",
                    "source_identity": "",
                    "redistribution_allowed": "unknown",
                    "derived_data_allowed": "unknown",
                    "confirmed_by_user": "",
                },
                "conversion_capability_prefill": {
                    "domain": "UCY",
                    "source_cv_feasible_after_terms": True,
                    "source_rows": [
                        {
                            "source_id": "UCY_zara01",
                            "trajectory_file": "zara01.txt",
                            "horizon_counts": {"50": 100, "100": 50},
                            "technical_conversion_ready_after_terms": True,
                            "causal_velocity_possible": True,
                            "central_velocity_used": False,
                            "blocked_by": ["terms/source_identity/path_version_not_confirmed"],
                        }
                    ],
                },
            }
        ]
    }

    rows = _source_plan_rows(intake)

    assert len(rows) == 1
    assert rows[0]["source_cv_feasible_after_terms_for_domain"] is True
    assert rows[0]["source_ready_now"] is False
    assert rows[0]["queued_now"] is False
    assert rows[0]["conversion_executed"] is False


def test_gf_gate_passes_for_safe_nonexecuting_plan():
    payload = {
        "source": "fresh_stage42_gf_post_confirmation_conversion_plan",
        "input_status": {"intake_exists": True, "ge_exists": True, "manifest_exists": True},
        "summary": {
            "planned_source_rows": 3,
            "technical_ready_after_terms_sources": 2,
            "source_cv_feasible_after_terms_datasets": 1,
            "t50_windows_after_terms": 100,
            "t100_windows_after_terms": 20,
            "source_ready_now": 0,
            "conversion_ready_targets_in_manifest": 0,
            "queued_now": 0,
            "downloaded_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
        },
        "user_action_required_written": True,
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gf_post_confirmation_conversion_plan_pass"
