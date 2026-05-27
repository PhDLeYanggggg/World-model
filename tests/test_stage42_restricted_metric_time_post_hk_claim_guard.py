from pathlib import Path

from src import stage42_restricted_metric_time_post_hk_claim_guard as hl
from src.stage14_pipeline import read_json


def _payload():
    path = Path("outputs/stage42_long_research/restricted_metric_time_post_hk_claim_guard_stage42.json")
    if path.exists():
        return read_json(path, {})
    return hl._build_payload()


def test_stage42_hl_scanner_flags_ready_now_overclaim(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.md"
    unsafe.write_text("ETH_UCY restricted metric/time ready now: true\n", encoding="utf-8")
    violations = hl.scan_file(unsafe)
    assert violations
    assert {row["check"] for row in violations} & {
        "eth_ucy_ready_now_overclaim",
        "restricted_metric_time_allowed_true",
    }


def test_stage42_hl_scanner_allows_boundary_context(tmp_path: Path) -> None:
    safe = tmp_path / "safe.md"
    safe.write_text(
        "ETH_UCY source support is technically repairable after terms, but ready now: `false`.\n",
        encoding="utf-8",
    )
    assert hl.scan_file(safe) == []


def test_stage42_hl_keeps_hk_ready_now_blocked() -> None:
    payload = _payload()
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    assert summary["hk_terms_confirmed"] is False
    assert summary["hk_ready_now"] is False
    assert summary["hk_conversion_ready_targets_now"] == 0
    assert claim["restricted_metric_time_claim_allowed_now"] is False
    assert claim["global_metric_claim_allowed"] is False
    assert claim["global_seconds_claim_allowed"] is False


def test_stage42_hl_gate_passes_without_conversion_claim() -> None:
    payload = _payload()
    gate = payload["stage42_hl_gate"]
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_hl_restricted_metric_time_post_hk_claim_guard_pass"
    assert payload["summary"]["download_executed"] is False
    assert payload["summary"]["conversion_executed"] is False
    assert payload["summary"]["evaluation_executed"] is False
    assert payload["summary"]["training_executed"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
