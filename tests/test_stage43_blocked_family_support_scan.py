from __future__ import annotations

from pathlib import Path

from src import stage43_blocked_family_support_scan as bc


def test_role_from_path_detects_raw_train_and_test() -> None:
    assert bc._role_from_path(Path("/tmp/TrajNet/Train/biwi/a.txt")) == "raw_train_dir"
    assert bc._role_from_path(Path("/tmp/TrajNet/Test/biwi/a.txt")) == "raw_test_dir"
    assert bc._role_from_path(Path("/tmp/TrajNet/Other/a.txt")) == "unknown_dir"


def test_parse_trajnet_txt_reports_horizon_windows(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    lines = [f"{i} 1 {float(i):.1f} 0.0" for i in range(60)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = bc._parse_trajnet_txt(path)
    assert out["parseable"] is True
    assert out["rows"] == 60
    assert out["track_count"] == 1
    assert out["horizon_window_counts"]["10"] == 50
    assert out["horizon_window_counts"]["50"] == 10
    assert out["horizon_window_counts"]["100"] == 0
    assert out["observation_step_horizon_window_counts"]["50"] == 10


def test_parse_trajnet_txt_counts_raw_frame_horizon_not_observation_steps(tmp_path: Path) -> None:
    path = tmp_path / "step10.txt"
    lines = [f"{i * 10} 1 {float(i):.1f} 0.0" for i in range(12)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = bc._parse_trajnet_txt(path)
    assert out["horizon_window_counts"]["50"] == 7
    assert out["observation_step_horizon_window_counts"]["50"] == 0


def test_family_summary_groups_roles_and_horizons() -> None:
    records = [
        {
            "family": "TrajNet_biwi",
            "path": "/tmp/Train/biwi/a.txt",
            "parseable": True,
            "raw_role": "raw_train_dir",
            "rows": 10,
            "track_count": 2,
            "horizon_window_counts": {"10": 5, "25": 0, "50": 0, "100": 0},
        },
        {
            "family": "TrajNet_biwi",
            "path": "/tmp/Test/biwi/b.txt",
            "parseable": True,
            "raw_role": "raw_test_dir",
            "rows": 20,
            "track_count": 3,
            "horizon_window_counts": {"10": 7, "25": 1, "50": 0, "100": 0},
        },
    ]
    out = bc._family_summary(records)
    assert out["TrajNet_biwi"]["file_count"] == 2
    assert out["TrajNet_biwi"]["raw_role_counts"] == {"raw_train_dir": 1, "raw_test_dir": 1}
    assert out["TrajNet_biwi"]["horizon_window_counts"]["10"] == 12


def test_blocked_family_action_keeps_repair_training_disallowed() -> None:
    action = bc._blocked_family_action(
        family="TrajNet_mot",
        bb_row={
            "ungated_improvement": -5.0,
            "split_support": {
                "train": {"family_rows": 0},
                "val": {"family_rows": 0},
                "test": {"family_rows": 10},
            },
        },
        raw_summary={
            "TrajNet_mot": {
                "raw_role_counts": {"raw_train_dir": 1},
                "horizon_window_counts": {"50": 0, "100": 0},
                "file_count": 1,
            }
        },
        min_validation_rows=1000,
    )
    assert action["repair_training_allowed_now"] is False
    assert "existing_ungated_transfer_catastrophic_negative" in action["blockers"]
    assert "single_source_family_no_independent_support_file" in action["blockers"]


def test_gate_passes_for_raw_support_scan_without_training_permission() -> None:
    payload = {
        "input_verdicts": {"stage43_bb": "stage43_bb_blocked_source_repair_feasibility_pass"},
        "summary": {
            "raw_file_count": 2,
            "parseable_raw_file_count": 2,
            "blocked_family_count": 2,
            "repair_training_allowed_now_count": 0,
        },
        "blocked_family_actions": [
            {
                "family": "TrajNet_biwi",
                "support_candidate_exists_in_raw_scan": True,
                "repair_training_allowed_now": False,
                "blockers": ["current_feature_store_has_no_train_family_rows"],
            },
            {
                "family": "TrajNet_mot",
                "support_candidate_exists_in_raw_scan": False,
                "repair_training_allowed_now": False,
                "blockers": ["single_source_family_no_independent_support_file"],
            },
        ],
        "next_required_actions": ["a", "b", "c"],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = bc._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_bc_blocked_family_support_scan_pass"
