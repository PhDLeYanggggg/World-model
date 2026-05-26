from __future__ import annotations

from pathlib import Path

from src import stage42_reviewer_replay_package as dm


def _gate_payload() -> dict:
    return {
        "inputs": {
            "evidence_provenance": {
                "stage42_cx_gate": {"passed": 20, "total": 20},
                "summary": {"artifacts_total": 25},
            },
            "paper_freeze_manifest": {
                "stage42_cz_gate": {"passed": 15, "total": 15},
                "freeze_status": {"freeze_status": "candidate_clean"},
                "manifest_hash": "a" * 64,
            },
            "proximity_batch_replay": {"stage42_cv_gate": {"passed": 25, "total": 25}},
            "group_consistency_replay": {"stage42_dk_gate": {"passed": 34, "total": 34}},
            "group_consistency_runtime": {
                "stage42_dl_gate": {"passed": 30, "total": 30},
                "real_batch_replay": {
                    "selected_xy_max_abs_diff": 0.0,
                    "selected_ade_max_abs_diff": 0.0,
                    "selected_fde_max_abs_diff": 0.0,
                    "switch_exact_match": True,
                    "metric": {
                        "all_improvement": 0.2,
                        "t50_improvement": 0.1,
                        "t100_raw_frame_diagnostic_improvement": 0.05,
                        "hard_failure_improvement": 0.2,
                    },
                    "diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01},
                },
            },
        },
        "required_files": [{"exists": True}],
        "commands_file": {"exists": True},
        "replay_commands": [
            ".venv-pytorch/bin/python run_stage42_replay_group_consistency_policy.py",
            ".venv-pytorch/bin/python -m pytest tests/test_stage42_reviewer_replay_package.py",
        ],
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_for_clean_reviewer_replay_payload() -> None:
    gate = dm._gate(_gate_payload())
    assert gate["verdict"] == "stage42_dm_reviewer_replay_package_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_training_command_in_replay_script() -> None:
    payload = _gate_payload()
    payload["replay_commands"] = [".venv-pytorch/bin/python run_stage42_train_hidden_model.py"]
    gate = dm._gate(payload)
    assert gate["gates"]["minimal_replay_has_no_training_commands"] is False
    assert gate["passed"] == gate["total"] - 1


def test_file_row_hashes_command_file(tmp_path: Path) -> None:
    path = tmp_path / "commands.sh"
    path.write_text(".venv-pytorch/bin/python run.py\n", encoding="utf-8")
    row = dm._file_row(path)
    assert row["exists"]
    assert row["size_bytes"] > 0
    assert len(row["sha256"]) == 64
