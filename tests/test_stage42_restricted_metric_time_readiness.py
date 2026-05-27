from pathlib import Path

from src import stage42_restricted_metric_time_readiness as hi
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/restricted_metric_time_readiness_stage42.json")
    if path.exists():
        return read_json(path, {})
    return hi._build_payload()


def test_stage42_hi_builds_eth_ucy_candidate_rows() -> None:
    payload = _payload()
    rows = payload["readiness_rows"]
    assert rows
    assert any(row["domain"] == "ETH_UCY" for row in rows)
    assert any(row["domain"] == "UCY" for row in rows)
    assert all(row["source_specific_metric_time_evidence"] for row in rows)


def test_stage42_hi_blocks_metric_time_claim_until_terms_ready() -> None:
    payload = _payload()
    summary = payload["summary"]
    boundary = payload["claim_boundary"]
    assert summary["technical_ready_after_terms_count"] >= 1
    assert summary["restricted_metric_time_ready_now_count"] == 0
    assert boundary["restricted_metric_time_claim_allowed_now"] is False
    assert boundary["global_metric_claim_allowed"] is False
    assert boundary["global_seconds_claim_allowed"] is False


def test_stage42_hi_gate_passes_with_terms_blocker() -> None:
    payload = _payload()
    gate = payload["stage42_hi_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_hi_restricted_metric_time_readiness_pass_blocked_by_terms"
    assert gate["gates"]["ready_now_zero"]
    assert gate["gates"]["paper_claim_blocked_now"]


def test_stage42_hi_user_action_points_to_validator() -> None:
    payload = _payload()
    actions = payload["user_action_required"]
    assert actions
    assert all("source_terms_confirmation_validator" in row["next_command"] for row in actions)
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
