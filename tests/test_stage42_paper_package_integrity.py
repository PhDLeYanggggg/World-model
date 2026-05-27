from src import stage42_paper_package_integrity as hx


def _payload() -> dict:
    return {
        "paper_files": [
            {"exists": True, "size_bytes": 2000, "sha256": "a" * 64, "source_label": "cached_verified", "path": "outputs/a.md"},
        ],
        "support_files": [
            {"exists": True, "size_bytes": 2000, "sha256": "b" * 64, "source_label": "cached_verified", "path": "outputs/b.md"},
        ],
        "objective_coverage": {
            key: {"present": True, "expected_status_present": True, "expected_status": spec["expected_status"], "source_label": "cached_verified"}
            for key, spec in hx.OBJECTIVE_ROWS.items()
        },
        "readiness_summary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
        },
        "paper_ready_evidence_matrix": "partial_blocked global_metric_seconds_claim_blocked",
        "replay_evidence_tiers": "T3_row_level_batch_replay",
        "reviewer_replay_package": "run_stage42_t100_runtime_row_cache_replay.py",
        "reviewer_replay_commands": "",
        "aggregate_paper_text": "true 3D foundation metric seconds Stage5C SMC raw-frame dataset-local",
        "readme_updates": {"readmes_updated": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_complete_integrity_payload() -> None:
    gate = hx._gate(_payload())
    assert gate["verdict"] == "stage42_hx_paper_package_integrity_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_missing_metric_blocker() -> None:
    payload = _payload()
    payload["paper_ready_evidence_matrix"] = "partial_blocked"
    gate = hx._gate(payload)
    assert gate["gates"]["metric_time_blocker_preserved"] is False


def test_objective_status_extracts_a_to_f_rows() -> None:
    text = "\n".join(f"{spec['marker']} {spec['expected_status']}" for spec in hx.OBJECTIVE_ROWS.values())
    rows = hx._extract_objective_status(text)
    assert set(rows) == set(hx.OBJECTIVE_ROWS)
    assert all(row["present"] and row["expected_status_present"] for row in rows.values())


def test_file_manifest_marks_missing_as_not_run(tmp_path) -> None:
    row = hx._file_manifest(tmp_path / "missing.md", "paper_deliverable")
    assert row["exists"] is False
    assert row["source_label"] == "not_run"
