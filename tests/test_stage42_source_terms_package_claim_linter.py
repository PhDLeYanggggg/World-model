from pathlib import Path

from src.stage42_source_terms_package_claim_linter import _gate, scan_file


def test_scan_flags_positive_opentraj_license_overclaim(tmp_path: Path):
    p = tmp_path / "bad.md"
    p.write_text("OpenTraj MIT permits ETH/UCY data reuse.\n", encoding="utf-8")
    violations = scan_file(p)
    assert violations
    assert violations[0]["check"] == "opentraj_license_overclaim"


def test_scan_allows_boundary_context_disclaimer(tmp_path: Path):
    p = tmp_path / "good.md"
    p.write_text("Do not write that OpenTraj MIT permits ETH/UCY data reuse.\n", encoding="utf-8")
    assert scan_file(p) == []


def test_scan_flags_auto_download_true(tmp_path: Path):
    p = tmp_path / "bad.md"
    p.write_text("auto-download allowed now: true\n", encoding="utf-8")
    violations = scan_file(p)
    assert violations
    assert violations[0]["check"] == "auto_download_overclaim"


def test_gate_passes_clean_package_lint():
    payload = {
        "source": "fresh_stage42_gq_source_terms_package_claim_linter",
        "input_status": {"go_exists": True, "gp_exists": True},
        "go_gate": {"passed": 14, "total": 14},
        "gp_gate": {"passed": 12, "total": 12},
        "summary": {
            "files_scanned": 10,
            "violation_count": 0,
            "underlying_data_license_confirmed": 0,
            "auto_download_allowed_now": 0,
            "contract_ready_now": 0,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "restricted_metric_time_claim_allowed_now": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required_written": True,
    }
    gate = _gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_gq_source_terms_package_claim_linter_pass"
