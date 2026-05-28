from __future__ import annotations

from pathlib import Path

from src import stage42_t50_source_specialist_reviewer_replay as s42in


def _gate_payload() -> dict:
    return {
        "inputs": {
            "stage42_ik": {"stage42_ik_gate": {"passed": 16, "total": 16}},
            "stage42_il": {"stage42_il_gate": {"passed": 16, "total": 16}},
            "stage42_im": {
                "stage42_im_gate": {"passed": 22, "total": 22},
                "replay": {"metric_summary_exact_replay": True},
                "frozen_policy": {
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoints_input": False,
                        "central_velocity": False,
                        "test_endpoint_goals": False,
                        "test_threshold_tuning": False,
                    }
                },
            },
        },
        "required_files": [
            {"exists": True, "sha256": "a" * 64},
            {"exists": True, "sha256": "b" * 64},
        ],
        "commands_file": {"exists": True, "sha256": "c" * 64},
        "replay_commands": [
            ".venv-pytorch/bin/python run_stage42_t50_ensemble_ucy_specialist_integration.py",
            ".venv-pytorch/bin/python run_stage42_t50_ucy_specialist_claim_audit.py",
        ],
        "reviewer_replay_summary": {
            "policy_hash": "d" * 64,
            "policy_artifact_sha256": "e" * 64,
            "ucy_t50_after": 0.12,
            "ucy_t50_delta": 0.12,
            "non_ucy_max_abs_delta": 1e-8,
            "ade_all": 0.15,
            "ade_t50": 0.10,
            "ade_hard_failure": 0.16,
            "ade_t50_ci_low": 0.09,
            "fde_t50_ci_low": 0.25,
            "ade_easy_degradation": 0.0,
            "supported_claims": ["claim a", "claim b", "claim c"],
            "blocked_claims": ["block a", "block b", "block c"],
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "independent_new_domain_claim": False,
            "source_specialist_replay_only": True,
        },
    }


def test_gate_passes_for_clean_reviewer_replay_payload() -> None:
    gate = s42in._gate(_gate_payload())
    assert gate["verdict"] == "stage42_in_t50_source_specialist_reviewer_replay_pass"
    assert gate["passed"] == gate["total"]


def test_gate_rejects_training_or_threshold_search_replay_commands() -> None:
    payload = _gate_payload()
    payload["replay_commands"] = [".venv-pytorch/bin/python run_stage42_train_threshold_search.py"]
    gate = s42in._gate(payload)
    assert gate["gates"]["no_training_or_threshold_search_commands"] is False
    assert gate["passed"] == gate["total"] - 1


def test_file_row_hashes_required_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")
    row = s42in._file_row(path)
    assert row["exists"] is True
    assert row["size_bytes"] > 0
    assert len(row["sha256"]) == 64


def test_real_run_records_source_specialist_boundaries() -> None:
    payload = s42in.run_stage42_t50_source_specialist_reviewer_replay()
    gate = payload["stage42_in_gate"]
    assert gate["verdict"] == "stage42_in_t50_source_specialist_reviewer_replay_pass"
    assert gate["gates"]["source_specialist_scope_only"] is True
    assert gate["gates"]["no_metric_seconds_overclaim"] is True
    assert gate["gates"]["stage5c_false"] is True
    assert gate["gates"]["smc_false"] is True
    assert payload["reviewer_replay_summary"]["ucy_t50_after"] > 0
