from __future__ import annotations

from src import stage42_ucy_h100_terms_prefill_from_integrity as gy


def test_prefill_row_carries_hash_but_not_legal_acceptance() -> None:
    template = {
        "dataset_id": "ucy_crowd_original",
        "candidate_id": "UCY_zara02::obsmat",
        "source_id": "UCY_zara02",
        "relative_path": "UCY/zara02/obsmat.txt",
    }
    integrity = {
        "source_identity_suggestion": "UCY::zara02::obsmat",
        "sha256": "a" * 64,
        "file_size_bytes": 123,
        "parsed_estimated_t100_windows": 42,
        "target_bucket_match": True,
        "t100_capable": True,
    }

    row = gy._prefill_row(template, integrity)

    assert row["file_sha256"] == "a" * 64
    assert row["source_identity_suggestion"] == "UCY::zara02::obsmat"
    assert row["terms_accepted_by_user"] is False
    assert row["allowed_use"] == ""
    assert row["local_path"] == ""
    assert row["confirmed_by_user"] == ""
    assert row["agent_may_fill_legal_acceptance"] is False


def test_summary_requires_legal_fields_to_stay_blank() -> None:
    prefill = {
        "datasets": [
            {
                "terms_accepted_by_user": False,
                "allowed_use": "",
                "local_path": "",
                "confirmed_by_user": "",
                "file_sha256": "a",
                "source_identity_suggestion": "x",
                "target_bucket_match": True,
                "t100_capable": True,
            }
        ]
    }
    gx = {"stage42_gx_gate": {"verdict": "stage42_gx_ucy_h100_candidate_integrity_manifest_pass"}}

    summary = gy._summary(prefill, gx)

    assert summary["legal_acceptance_fields_blank"] is True
    assert summary["conversion_ready_now_count"] == 0


def test_gate_passes_for_prefill_payload() -> None:
    payload = {
        "summary": {
            "input_gx_verdict": "stage42_gx_ucy_h100_candidate_integrity_manifest_pass",
            "prefill_rows": 2,
            "rows_with_hash": 2,
            "rows_with_source_identity_suggestion": 2,
            "target_family_rows": 1,
            "legal_acceptance_fields_blank": True,
            "conversion_ready_now_count": 0,
            "downloaded_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
        },
        "prefill": {
            "agent_may_fill_legal_acceptance": False,
            "datasets": [
                {"agent_may_fill_legal_acceptance": False},
                {"agent_may_fill_legal_acceptance": False},
            ],
        },
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = gy._gate(payload)

    assert gate["verdict"] == "stage42_gy_ucy_h100_terms_prefill_pass"
    assert gate["passed"] == gate["total"]
