from __future__ import annotations

from pathlib import Path

from src import stage42_runtime_replay_paper_refresh as cw


def test_refresh_lines_keep_replay_and_claim_boundaries() -> None:
    summary = {
        "policy_hash": "abc",
        "gate": {"passed": 25, "total": 25, "verdict": "ok"},
        "val_rows": 3,
        "test_rows": 4,
        "val_decision_exact_replay": True,
        "test_decision_exact_replay": True,
        "test_selected_xy_max_abs_diff": 0.0,
        "test_selected_ade_max_abs_diff": 0.0,
        "test_selected_fde_max_abs_diff": 0.0,
        "test_vs_endpoint_linear_ade": {
            "all_improvement": 0.01,
            "t50_improvement": 0.02,
            "t100_raw_frame_diagnostic_improvement": 0.03,
            "hard_failure_improvement": 0.04,
            "easy_degradation": 0.001,
            "switch_rate": 0.2,
        },
        "joint_safety_vs_endpoint_linear": {
            "near_collision_002_delta": 0.0,
            "near_collision_005_delta": -0.001,
            "p05_min_group_distance_delta": 0.0,
            "jagged_rate_delta": 0.0,
        },
    }
    text = "\n".join(cw._refresh_lines(summary))
    assert "exact runtime replay" in text
    assert "not future labels" in text
    assert "does not create a new metric/seconds/3D/foundation claim" in text
    assert "Stage5C remains unexecuted and SMC remains disabled" in text


def test_refresh_paper_files_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    paper = tmp_path / "method.md"
    paper.write_text("# Method\n", encoding="utf-8")
    monkeypatch.setattr(cw, "PAPER_FILES", [paper])
    summary = {
        "policy_hash": "abc",
        "gate": {"passed": 25, "total": 25, "verdict": "ok"},
        "val_rows": 3,
        "test_rows": 4,
        "val_decision_exact_replay": True,
        "test_decision_exact_replay": True,
        "test_selected_xy_max_abs_diff": 0.0,
        "test_selected_ade_max_abs_diff": 0.0,
        "test_selected_fde_max_abs_diff": 0.0,
        "test_vs_endpoint_linear_ade": {
            "all_improvement": 0.01,
            "t50_improvement": 0.02,
            "t100_raw_frame_diagnostic_improvement": 0.03,
            "hard_failure_improvement": 0.04,
            "easy_degradation": 0.001,
            "switch_rate": 0.2,
        },
        "joint_safety_vs_endpoint_linear": {
            "near_collision_002_delta": 0.0,
            "near_collision_005_delta": -0.001,
            "p05_min_group_distance_delta": 0.0,
            "jagged_rate_delta": 0.0,
        },
    }
    first = cw._refresh_paper_files(summary)
    second = cw._refresh_paper_files(summary)
    text = paper.read_text(encoding="utf-8")
    assert first[0]["contains_stage42_cw"]
    assert second[0]["contains_stage42_cw"]
    assert text.count("STAGE42_CW_RUNTIME_REPLAY_REFRESH:START") == 1
    assert "Stage42-CW Runtime Replay" in text


def test_gate_requires_exact_replay_and_boundaries() -> None:
    payload = {
        "source": "fresh_synthesis_from_stage42_cv_runtime_batch_replay",
        "inputs": {
            "stage42_ct": {"stage42_ct_gate": {"passed": 30, "total": 30}},
            "stage42_cu": {"stage42_cu_gate": {"passed": 19, "total": 19}},
        },
        "paper_file_status": [{"contains_stage42_cw": True}],
        "runtime_replay_summary": {
            "gate": {"passed": 25, "total": 25},
            "val_decision_exact_replay": True,
            "test_decision_exact_replay": True,
            "test_selected_xy_max_abs_diff": 0.0,
            "test_selected_ade_max_abs_diff": 0.0,
            "test_selected_fde_max_abs_diff": 0.0,
            "test_vs_endpoint_linear_ade": {
                "all_improvement": 0.01,
                "t50_improvement": 0.01,
                "t100_raw_frame_diagnostic_improvement": 0.01,
                "hard_failure_improvement": 0.01,
                "easy_degradation": 0.001,
            },
            "joint_safety_vs_endpoint_linear": {"near_collision_005_delta": -0.001},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cw._gate(payload)
    assert gate["verdict"] == "stage42_cw_runtime_replay_paper_refresh_pass"
    assert gate["passed"] == gate["total"]
