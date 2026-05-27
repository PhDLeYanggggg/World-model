from __future__ import annotations

from src import stage42_unified_guarded_conversion_queue as ft


def test_h100_queue_rows_are_preserved_as_nonexecuting() -> None:
    queue = ft._h100_queue(
        {
            "queue": [
                {
                    "candidate_id": "UCY_zara02::obsmat",
                    "source_id": "UCY_zara02",
                    "candidate_file": "/tmp/ucy/UCY/zara02/obsmat.txt",
                    "relative_path": "UCY/zara02/obsmat.txt",
                    "estimated_t100_windows": 2095,
                    "guarded_conversion_next_checks": ["no future endpoint inference input"],
                }
            ]
        }
    )

    assert len(queue) == 1
    assert queue[0]["queue_type"] == "ucy_h100_candidate"
    assert queue[0]["conversion_executed"] is False
    assert queue[0]["evaluation_executed"] is False


def test_global_queue_rows_are_preserved_as_nonexecuting() -> None:
    queue = ft._global_queue(
        {
            "conversion_ready_targets": [
                {
                    "dataset_id": "eth_biwi_original",
                    "official_url": "https://example.edu/eth",
                    "confirmed_local_path": "/tmp/eth",
                    "source_identity": "ETH_seq_eth",
                    "conversion_ready": True,
                }
            ]
        }
    )

    assert len(queue) == 1
    assert queue[0]["queue_type"] == "source_conversion_ready_target"
    assert queue[0]["conversion_executed"] is False
    assert queue[0]["evaluation_executed"] is False


def test_gate_passes_for_empty_queue_with_blocked_actions() -> None:
    payload = {
        "source": ft.SOURCE,
        "summary": {
            "source_manifest_source": "fresh_stage42_cg_source_terms_confirmation_validator",
            "fs_verdict": "stage42_fs_ucy_h100_terms_intake_validator_pass",
            "unified_queue_count": 0,
            "source_queue_count": 0,
            "h100_queue_count": 0,
            "blocked_action_count": 1,
            "downloaded_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
        },
        "unified_queue": [],
        "blocked_actions": [{"dataset_id": "ucy"}],
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
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = ft._gate(payload)

    assert gate["verdict"] == "stage42_ft_unified_guarded_conversion_queue_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_empty_queue_has_no_blocked_actions() -> None:
    payload = {
        "source": ft.SOURCE,
        "summary": {
            "source_manifest_source": "fresh_stage42_cg_source_terms_confirmation_validator",
            "fs_verdict": "stage42_fs_ucy_h100_terms_intake_validator_pass",
            "unified_queue_count": 0,
            "source_queue_count": 0,
            "h100_queue_count": 0,
            "blocked_action_count": 0,
            "downloaded_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
        },
        "unified_queue": [],
        "blocked_actions": [],
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
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = ft._gate(payload)

    assert gate["gates"]["blocked_actions_preserved_when_empty"] is False
    assert gate["verdict"] == "stage42_ft_unified_guarded_conversion_queue_partial"
