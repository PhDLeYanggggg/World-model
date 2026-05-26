from __future__ import annotations

from pathlib import Path

from src import stage42_source_support_closure_audit as dd


def _payloads() -> dict:
    return {
        "source_terms": {
            "source": "fresh_stage42_cg_source_terms_confirmation_validator",
            "summary": {"conversion_ready_targets": 0},
        },
        "time_geometry": {
            "source": "fresh_source_time_geometry_calibration_audit",
            "source_records": [
                {
                    "source_id": "ETH_seq_eth",
                    "domain": "ETH_UCY",
                    "local_claim": "source_specific_annotation_step_meter_coordinate_evidence",
                    "global_metric_claim_allowed": False,
                    "global_seconds_claim_allowed": False,
                },
                {
                    "source_id": "UCY_zara01",
                    "domain": "UCY",
                    "local_claim": "source_specific_annotation_step_meter_coordinate_evidence",
                    "global_metric_claim_allowed": False,
                    "global_seconds_claim_allowed": False,
                },
            ],
        },
        "t100_gap": {
            "source": "fresh_synthesis_from_stage42_ba_and_calibration",
            "summary": {
                "unsupported_t100_domains": ["ETH_UCY", "TrajNet", "UCY"],
                "additional_t100_sources_needed_by_domain": {"ETH_UCY": 2, "TrajNet": 1, "UCY": 1},
            },
        },
        "conversion_manifest": {
            "source": "fresh_stage42_cg_source_terms_confirmation_validator",
            "conversion_ready_targets": [],
        },
        "source_diversity_preflight": {
            "source": "fresh_stage42_ce_source_diversity_conversion_preflight",
            "target_summaries": [
                {"target": "eth_biwi_original", "t50_files": 3, "t100_files": 2, "legal_blocked": True},
                {"target": "trajnetplusplus_official", "t50_files": 0, "t100_files": 0, "legal_blocked": True},
                {"target": "ucy_crowd_original", "t50_files": 6, "t100_files": 6, "legal_blocked": True},
                {"target": "opentraj_toolkit", "t50_files": 10, "t100_files": 10, "legal_blocked": False},
            ],
        },
        "local_t100_schema": {
            "source": "fresh_in_memory_schema_conversion",
            "summary": {
                "source_cv_domains_evaluated": ["ETH_UCY", "UCY"],
                "source_cv_domains_positive_vs_constant_velocity": ["UCY"],
            },
        },
        "t100_source_cv": {
            "source": "fresh_run",
            "summary": {"supported_t100_domains": []},
        },
    }


def test_domain_status_keeps_global_claims_blocked() -> None:
    payloads = _payloads()
    eth = dd._domain_status("ETH_UCY", payloads)
    traj = dd._domain_status("TrajNet", payloads)
    ucy = dd._domain_status("UCY", payloads)
    assert eth["claim_status"] == "not_closed"
    assert "source_terms_confirmation_or_conversion_readiness_missing" in eth["blockers"]
    assert "source_specific_metric_time_calibration_missing" in traj["blockers"]
    assert ucy["partial_support"]["local_t100_schema_positive_vs_constant_velocity"] is True
    assert ucy["global_t100_deployable_claim_allowed"] is False


def test_gate_passes_with_open_blockers_and_user_action() -> None:
    result = {
        "input_status": {
            key: {"exists": True, "source": value["source"], "generated_at_utc": ""}
            for key, value in _payloads().items()
        },
        "domain_status": [dd._domain_status(domain, _payloads()) for domain in dd.DOMAINS],
        "closure_summary": {
            "restricted_source_specific_metric_time_candidate_exists": True,
        },
        "user_action_required": [{"domain": "ETH_UCY"}],
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "global_t100_deployable_claim_allowed": False,
            "converted_dataset_claim_from_stage42_dd": False,
            "evaluation_claim_from_stage42_dd": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = dd._gate(result)
    assert gate["verdict"] == "stage42_dd_source_support_closure_audit_pass_open_blockers"
    assert gate["passed"] == gate["total"]


def test_run_writes_isolated_outputs(tmp_path: Path, monkeypatch) -> None:
    paths = {key: tmp_path / f"{key}.json" for key in dd.INPUTS}
    for key, path in paths.items():
        import json

        path.write_text(json.dumps(_payloads()[key]), encoding="utf-8")
    monkeypatch.setattr(dd, "INPUTS", paths)
    monkeypatch.setattr(dd, "REPORT_JSON", tmp_path / "source_support_closure_audit_stage42.json")
    monkeypatch.setattr(dd, "REPORT_MD", tmp_path / "source_support_closure_audit_stage42.md")
    monkeypatch.setattr(dd, "GATE_MD", tmp_path / "stage42_stage_dd_gate.md")
    monkeypatch.setattr(dd, "USER_ACTION_MD", tmp_path / "user_action_required_source_support_closure_stage42.md")
    result = dd.run_stage42_source_support_closure_audit(refresh_readmes=False)
    assert result["stage42_dd_gate"]["verdict"] == "stage42_dd_source_support_closure_audit_pass_open_blockers"
    assert dd.REPORT_JSON.exists()
    assert dd.REPORT_MD.exists()
    assert dd.USER_ACTION_MD.exists()
