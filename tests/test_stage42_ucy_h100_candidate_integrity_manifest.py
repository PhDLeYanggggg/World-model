from __future__ import annotations

from pathlib import Path

from src import stage42_ucy_h100_candidate_integrity_manifest as gx


def test_source_identity_suggestion_is_stable_for_ucy_paths() -> None:
    assert gx._source_identity_suggestion("UCY/zara02/obsmat.txt") == "UCY::zara02::obsmat"
    assert gx._source_identity_suggestion("UCY/students03/students003.txt") == "UCY::students03::students003"


def test_resolve_candidate_path_uses_opentraj_dataset_root() -> None:
    path = gx._resolve_candidate_path("UCY/zara02/obsmat.txt")

    assert path == Path("external_data/OpenTraj/datasets/UCY/zara02/obsmat.txt")


def test_gate_passes_for_integrity_manifest_payload() -> None:
    payload = {
        "summary": {
            "input_gw_verdict": "stage42_gw_h100_blocker_closure_decision_pass",
            "input_fq_verdict": "stage42_fq_h100_source_support_repair_queue_pass",
            "input_fs_verdict": "stage42_fs_ucy_h100_terms_intake_validator_pass",
            "candidate_rows": 2,
            "existing_files": 2,
            "target_family_candidates": 1,
            "t100_capable_files": 2,
            "total_parsed_rows": 100,
            "unique_hashes": 2,
            "conversion_ready_now_count": 0,
            "downloaded_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
            "ucy_gw_legal_conversion_ready": False,
        },
        "candidate_integrity_rows": [
            {
                "sha256": "a" * 64,
                "raw_content_stored_in_manifest": False,
            },
            {
                "sha256": "b" * 64,
                "raw_content_stored_in_manifest": False,
            },
        ],
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

    gate = gx._gate(payload)

    assert gate["verdict"] == "stage42_gx_ucy_h100_candidate_integrity_manifest_pass"
    assert gate["passed"] == gate["total"]
