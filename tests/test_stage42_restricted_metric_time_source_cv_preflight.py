from pathlib import Path

from src import stage42_restricted_metric_time_source_cv_preflight as hj
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/restricted_metric_time_source_cv_preflight_stage42.json")
    if path.exists():
        return read_json(path, {})
    return hj._build_payload()


def test_stage42_hj_parses_candidate_sources() -> None:
    payload = _payload()
    rows = payload["source_rows"]
    assert rows
    assert all(row["track_stats"]["rows"] > 0 for row in rows)
    assert any(row["t50_windows"] > 0 for row in rows)
    assert any(row["t100_windows"] > 0 for row in rows)
    assert any(row["source_id"] == "ETH_seq_hotel" and row["t100_windows"] == 0 for row in rows)


def test_stage42_hj_source_cv_feasible_after_terms() -> None:
    payload = _payload()
    summary = payload["summary"]
    assert "ETH_UCY" in summary["domains_source_cv_blocked_after_terms"]
    assert "UCY" in summary["domains_source_cv_feasible_after_terms"]
    assert "UCY" in summary["domains_robust_source_cv_feasible_after_terms"]
    assert summary["restricted_metric_time_ready_now_sources"] == 0


def test_stage42_hj_gate_passes_but_blocks_claim() -> None:
    payload = _payload()
    gate = payload["stage42_hj_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_hj_restricted_metric_time_source_cv_preflight_pass_with_eth_ucy_source_cv_limit"
    assert payload["claim_boundary"]["source_cv_preflight_is_conversion"] is False
    assert payload["claim_boundary"]["restricted_metric_time_claim_allowed_now"] is False
    assert payload["claim_boundary"]["global_metric_claim_allowed"] is False


def test_stage42_hj_user_action_keeps_stage5c_smc_false() -> None:
    payload = _payload()
    assert payload["user_action_required"]
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
