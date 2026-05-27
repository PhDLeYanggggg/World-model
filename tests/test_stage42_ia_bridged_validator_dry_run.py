from pathlib import Path

from src.stage42_ia_bridged_validator_dry_run import _gate


def _payload(**summary_overrides):
    summary = {
        "targets_validated": 5,
        "conversion_ready_targets": 0,
        "converted_datasets_now": 0,
        "evaluated_datasets_now": 0,
        "canonical_manifest_overwritten": False,
    }
    summary.update(summary_overrides)
    return {
        "input_reports": {
            "stage42_cf_verdict": "stage42_cf_source_conversion_legal_gate_pass",
            "bridged_template_source": "fresh_stage42_ia_hz_to_cg_intake_bridge",
            "active_validator_input": False,
        },
        "summary": summary,
        "readme_updates": {"readmes_updated": True, "paper_matrix_updated": True},
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _patch_existing_paths(monkeypatch, tmp_path: Path):
    from src import stage42_ia_bridged_validator_dry_run as mod

    paths = {
        "IA_BRIDGED_TEMPLATE_JSON": tmp_path / "bridged.json",
        "MANIFEST_JSON": tmp_path / "manifest.json",
        "MANIFEST_MD": tmp_path / "manifest.md",
        "USER_ACTION_MD": tmp_path / "user_action.md",
    }
    for name, path in paths.items():
        path.write_text("x", encoding="utf-8")
        monkeypatch.setattr(mod, name, path)


def test_gate_passes_for_blank_inactive_bridge(monkeypatch, tmp_path):
    _patch_existing_paths(monkeypatch, tmp_path)
    gate = _gate(_payload())
    assert gate["verdict"] == "stage42_ib_ia_bridged_validator_dry_run_pass"
    assert gate["passed"] == gate["total"]


def test_gate_fails_if_bridge_is_active(monkeypatch, tmp_path):
    _patch_existing_paths(monkeypatch, tmp_path)
    payload = _payload()
    payload["input_reports"]["active_validator_input"] = True
    gate = _gate(payload)
    assert gate["gates"]["bridged_template_inactive"] is False
    assert gate["verdict"] == "stage42_ib_ia_bridged_validator_dry_run_partial"


def test_gate_fails_if_blank_bridge_claims_ready(monkeypatch, tmp_path):
    _patch_existing_paths(monkeypatch, tmp_path)
    gate = _gate(_payload(conversion_ready_targets=1))
    assert gate["gates"]["blank_hz_blocks_conversion"] is False
    assert gate["verdict"] == "stage42_ib_ia_bridged_validator_dry_run_partial"


def test_gate_fails_on_metric_overclaim(monkeypatch, tmp_path):
    _patch_existing_paths(monkeypatch, tmp_path)
    payload = _payload()
    payload["claim_boundary"]["metric_or_seconds_claim"] = True
    gate = _gate(payload)
    assert gate["gates"]["no_metric_seconds_claim"] is False
