from __future__ import annotations

from pathlib import Path

from src import stage42_ucy_h100_terms_intake_validator as fs
from src import stage42_ucy_h100_terms_gated_conversion_preflight as fr


def _candidate(**overrides):
    row = {
        "candidate_id": "UCY_zara02::obsmat",
        "source_id": "UCY_zara02",
        "relative_path": "UCY/zara02/obsmat.txt",
        "target_bucket_match": True,
        "estimated_t100_windows": 2095,
    }
    row.update(overrides)
    return row


def _template_row(local_path: str = "", **overrides):
    row = {
        "candidate_id": "UCY_zara02::obsmat",
        "source_id": "UCY_zara02",
        "relative_path": "UCY/zara02/obsmat.txt",
        "official_terms_url": fr.UCY_OFFICIAL_URL,
        "terms_accepted_by_user": True,
        "terms_acceptance_date": "2026-05-27",
        "accepted_terms_version_or_access_date": "2026-05-27",
        "allowed_use": "research_only",
        "redistribution_allowed": "no",
        "derived_data_allowed": "yes",
        "local_path": local_path,
        "source_identity": "UCY_zara02 official local source",
        "confirmed_by_user": "yangyue",
        "agent_may_fill": False,
    }
    row.update(overrides)
    return row


def test_blank_or_unconfirmed_template_row_blocks_conversion() -> None:
    validation = fs._validate_template_row({"candidate_id": "UCY_zara02::obsmat"}, _candidate())

    assert validation["terms_intake_ready"] is False
    assert "terms_not_accepted" in validation["blockers"]
    assert "local_path_confirmation_missing" in validation["blockers"]
    assert "confirmed_by_user_missing" in validation["blockers"]


def test_ready_row_requires_candidate_file_and_user_confirmation(tmp_path: Path) -> None:
    candidate_file = tmp_path / "UCY" / "zara02" / "obsmat.txt"
    candidate_file.parent.mkdir(parents=True)
    candidate_file.write_text("0 1 0.0 0.0\n")

    validation = fs._validate_template_row(_template_row(str(tmp_path)), _candidate())

    assert validation["terms_intake_ready"] is True
    assert validation["conversion_queue_eligible"] is True
    assert validation["candidate_file"] == str(candidate_file)


def test_guarded_queue_never_claims_conversion(tmp_path: Path) -> None:
    candidate_file = tmp_path / "UCY" / "zara02" / "obsmat.txt"
    candidate_file.parent.mkdir(parents=True)
    candidate_file.write_text("0 1 0.0 0.0\n")
    ready = fs._validate_template_row(_template_row(str(tmp_path)), _candidate())
    blocked = fs._validate_template_row({"candidate_id": "missing"}, None)

    queue = fs._guarded_conversion_queue([ready, blocked])

    assert len(queue) == 1
    assert queue[0]["conversion_executed"] is False
    assert queue[0]["evaluation_executed"] is False
    assert "no future endpoint inference input" in queue[0]["guarded_conversion_next_checks"]


def test_gate_passes_with_empty_queue_when_all_candidates_are_legally_blocked() -> None:
    payload = {
        "source": fs.SOURCE,
        "summary": {
            "input_fr_verdict": "stage42_fr_ucy_h100_terms_gated_preflight_pass",
            "candidate_rows_validated": 1,
            "target_family_candidates": 1,
            "terms_ready_candidates": 0,
            "guarded_conversion_queue_count": 0,
            "downloaded_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
        },
        "input_reports": {"template_source": fr.SOURCE},
        "validations": [
            {
                "terms_intake_ready": False,
                "blockers": ["terms_not_accepted"],
            }
        ],
        "guarded_conversion_queue": [],
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "uniform_horizon_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fs._gate(payload)

    assert gate["verdict"] == "stage42_fs_ucy_h100_terms_intake_validator_pass"
    assert gate["passed"] == gate["total"]
