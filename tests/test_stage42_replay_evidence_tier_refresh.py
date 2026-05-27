from __future__ import annotations

from pathlib import Path

from src import stage42_replay_evidence_tier_refresh as hw


def _payload() -> dict:
    return {
        "inputs": {
            "stage42_hs": {"stage42_hs_gate": {"passed": 27, "total": 27}},
            "stage42_ht": {"stage42_ht_gate": {"passed": 19, "total": 19}},
            "stage42_hu": {
                "stage42_hu_gate": {"passed": 17, "total": 17},
                "sufficiency": {"real_batch_replay_status": "not_run", "blocker": "missing_row_level_candidate_floor_selected_arrays"},
            },
            "stage42_hv": {
                "stage42_hv_gate": {"passed": 28, "total": 28},
                "runtime_batch_replay": {
                    "rows": 10,
                    "t100_rows": 3,
                    "metric": {
                        "all_improvement": 0.2,
                        "t50_improvement": 0.1,
                        "t100_raw_frame_diagnostic_improvement": 0.05,
                        "hard_failure_improvement": 0.2,
                        "easy_degradation": -0.1,
                        "t100_easy_degradation": -0.01,
                    },
                },
            },
        },
        "evidence_tiers": [
            {"tier": "T0_artifact_presence", "status": "pass"},
            {"tier": "T1_runtime_smoke_replay", "status": "pass"},
            {"tier": "T2_frozen_metric_replay", "status": "pass"},
            {"tier": "T2_5_blocker_audit", "status": "resolved_by_hv"},
            {"tier": "T3_row_level_batch_replay", "status": "pass", "rows": 10, "cache_committed": False},
        ],
        "commands_file": {"exists": True},
        "replay_commands": [
            ".venv-pytorch/bin/python run_stage42_t100_runtime_row_cache_replay.py",
            ".venv-pytorch/bin/python -m pytest tests/test_stage42_replay_evidence_tier_refresh.py",
        ],
        "paper_updates": {"paper_matrix_updated": True, "reviewer_package_updated": True, "readmes_updated": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_for_complete_tier_payload() -> None:
    gate = hw._gate(_payload())
    assert gate["verdict"] == "stage42_hw_replay_evidence_tier_refresh_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_training_command() -> None:
    payload = _payload()
    payload["replay_commands"] = [".venv-pytorch/bin/python run_stage42_train_forbidden.py"]
    gate = hw._gate(payload)
    assert gate["gates"]["commands_do_not_train_or_execute_forbidden"] is False
    assert gate["passed"] == gate["total"] - 1


def test_tier_table_includes_row_level_batch_replay() -> None:
    table = "\n".join(hw._tier_table(_payload()["evidence_tiers"]))
    assert "T3_row_level_batch_replay" in table
    assert "pass" in table


def test_file_row_hashes_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf-8")
    row = hw._file_row(path)
    assert row["exists"] is True
    assert row["size_bytes"] == 2
    assert len(row["sha256"]) == 64
