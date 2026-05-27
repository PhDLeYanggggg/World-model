from pathlib import Path

from src import stage42_restricted_metric_time_conversion_queue_v2 as hn
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/restricted_metric_time_conversion_queue_v2_stage42.json")
    if path.exists():
        return read_json(path, {})
    return hn._build_payload()


def test_stage42_hn_ready_candidate_queue_is_nonexecuting() -> None:
    queue = hn.queue_from_ready_candidates(
        {
            "ready_candidates": [
                {
                    "candidate_id": "hj::UCY_zara02",
                    "source_id": "UCY_zara02",
                    "domain": "UCY",
                    "terms_target_id": "ucy_crowd_original",
                    "conversion_ready": True,
                    "source_cv_usable_after_terms": True,
                    "t50_windows_after_terms": 100,
                    "t100_windows_after_terms": 50,
                }
            ]
        }
    )
    assert len(queue) == 1
    row = queue[0]
    assert row["queue_type"] == "restricted_metric_time_source_candidate"
    assert row["conversion_executed"] is False
    assert row["evaluation_executed"] is False
    assert "no future endpoint input" in row["required_next_checks"]


def test_stage42_hn_blocked_actions_preserve_terms_blockers() -> None:
    blocked = hn.blocked_actions_from_manifest(
        {
            "blocked_candidates": [
                {
                    "candidate_id": "hk::ETH-Person_bahnhof_assc_gt",
                    "source_id": "ETH-Person_bahnhof_assc_gt",
                    "domain": "ETH_UCY",
                    "terms_target_id": "eth_person_local_candidates",
                    "t50_windows_after_terms": 1897,
                    "t100_windows_after_terms": 348,
                    "blockers": ["terms_not_accepted_by_user", "official_terms_url_missing"],
                }
            ]
        }
    )
    assert len(blocked) == 1
    assert blocked[0]["queued"] is False
    assert "terms_not_accepted_by_user" in blocked[0]["blockers"]


def test_stage42_hn_current_manifest_refuses_conversion() -> None:
    payload = _payload()
    summary = payload["summary"]
    assert summary["manifest_ready_candidates"] == 0
    assert summary["conversion_queue_count"] == 0
    assert summary["blocked_action_count"] > 0
    assert summary["conversion_executed"] is False
    assert summary["evaluation_executed"] is False
    assert payload["claim_boundary"]["restricted_metric_time_claim_allowed_now"] is False


def test_stage42_hn_gate_passes_blocked_until_ready_candidates() -> None:
    payload = _payload()
    gate = payload["stage42_hn_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_hn_restricted_metric_time_conversion_queue_v2_pass_blocked_until_ready_candidates"
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
