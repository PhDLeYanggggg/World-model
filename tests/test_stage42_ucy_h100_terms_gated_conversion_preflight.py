from __future__ import annotations

from src import stage42_ucy_h100_terms_gated_conversion_preflight as fr


def test_source_id_maps_ucy_relative_paths() -> None:
    assert fr._source_id("UCY/zara02/obsmat.txt") == "UCY_zara02"
    assert fr._source_id("UCY/students03/obsmat_px.txt") == "UCY_students03"


def test_confirmation_blockers_preserve_unconfirmed_terms() -> None:
    blockers = fr._confirmation_blockers({})

    assert "terms_not_accepted" in blockers
    assert "local_path_confirmation_missing" in blockers
    assert "source_identity_missing" in blockers
    assert "official_terms_url_mismatch_or_missing" in blockers


def test_candidate_rows_are_sorted_by_target_family_and_windows() -> None:
    fq_payload = {
        "key_rows": {
            "UCY|100": {
                "top_candidates": [
                    {
                        "relative_path": "UCY/students03/obsmat.txt",
                        "family_bucket": "students",
                        "target_bucket_match": False,
                        "max_track_points": 539,
                        "estimated_t100_windows": 3413,
                    },
                    {
                        "relative_path": "UCY/zara02/obsmat.txt",
                        "family_bucket": "zara",
                        "target_bucket_match": True,
                        "max_track_points": 583,
                        "estimated_t100_windows": 2095,
                    },
                ]
            }
        }
    }

    rows = fr._candidate_rows(fq_payload, {})

    assert rows[0]["source_id"] == "UCY_zara02"
    assert rows[0]["target_bucket_match"] is True
    assert rows[0]["conversion_preflight_ready"] is False
    assert rows[1]["source_id"] == "UCY_students03"


def test_gate_passes_when_terms_block_conversion_without_overclaim() -> None:
    payload = {
        "source": fr.SOURCE,
        "summary": {
            "input_fq_verdict": "stage42_fq_h100_source_support_repair_queue_pass",
            "candidate_rows": 2,
            "target_family_candidates": 1,
            "conversion_preflight_ready_count": 0,
            "conversion_queue_count": 0,
            "downloaded_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
            "terms_confirmation_blockers": ["terms_not_accepted"],
        },
        "template_written": True,
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "converted_dataset_claim_allowed": False,
            "uniform_horizon_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fr._gate(payload)

    assert gate["verdict"] == "stage42_fr_ucy_h100_terms_gated_preflight_pass"
    assert gate["passed"] == gate["total"]
