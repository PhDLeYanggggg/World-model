from __future__ import annotations

from src.stage42_guarded_source_conversion_launcher import (
    _blocked_actions,
    _build_conversion_queue,
    _gate,
    _summary,
)


def _blocked_manifest() -> dict:
    return {
        "source": "fresh_stage42_cg_source_terms_confirmation_validator",
        "generated_at_utc": "2026-05-27T00:00:00+00:00",
        "conversion_ready_targets": [],
        "blocked_targets": [
            {
                "dataset_id": "ucy_crowd_original",
                "name": "UCY Crowd",
                "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
                "cf_blockers": ["manual_terms_or_application_required"],
                "confirmation_blockers": ["terms_not_accepted", "local_path_confirmation_missing"],
                "conversion_ready": False,
                "next_action": "fill confirmation",
            }
        ],
        "conversion_executed": False,
        "evaluation_executed": False,
    }


def _ready_manifest() -> dict:
    return {
        "source": "synthetic_ready_manifest_for_unit_test",
        "generated_at_utc": "2026-05-27T00:00:00+00:00",
        "conversion_ready_targets": [
            {
                "dataset_id": "ucy_crowd_original",
                "name": "UCY Crowd",
                "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
                "confirmed_local_path": "/tmp/ucy",
                "source_identity": "official_ucy_source_confirmed_by_user",
                "conversion_ready": True,
            }
        ],
        "blocked_targets": [],
        "conversion_executed": False,
        "evaluation_executed": False,
    }


def _payload(manifest: dict) -> dict:
    queue = _build_conversion_queue(manifest)
    blocked = _blocked_actions(manifest)
    return {
        "input_manifest_hash": "hash",
        "summary": _summary(manifest, queue, blocked),
        "conversion_queue": queue,
        "blocked_actions": blocked,
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required_written": True,
    }


def test_stage42_ej_blank_manifest_refuses_conversion() -> None:
    payload = _payload(_blocked_manifest())

    assert payload["summary"]["ready_targets_in_manifest"] == 0
    assert payload["summary"]["conversion_queue_count"] == 0
    assert payload["summary"]["conversion_executed"] is False
    assert payload["summary"]["evaluation_executed"] is False
    assert payload["blocked_actions"][0]["dataset_id"] == "ucy_crowd_original"

    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ej_guarded_source_conversion_launcher_pass"


def test_stage42_ej_ready_manifest_queues_but_does_not_execute() -> None:
    payload = _payload(_ready_manifest())

    assert payload["summary"]["ready_targets_in_manifest"] == 1
    assert payload["summary"]["conversion_queue_count"] == 1
    assert payload["conversion_queue"][0]["status"] == "queued_for_future_guarded_conversion"
    assert payload["conversion_queue"][0]["execution_in_stage42_ej"] is False
    assert payload["summary"]["conversion_executed"] is False
    assert payload["summary"]["evaluation_executed"] is False

    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
