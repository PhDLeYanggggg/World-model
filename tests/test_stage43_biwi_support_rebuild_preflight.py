from __future__ import annotations

import numpy as np

from src import stage43_biwi_support_rebuild_preflight as bd


def test_biwi_source_detection_and_role() -> None:
    train = "/x/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt"
    test = "/x/OpenTraj/datasets/TrajNet/Test/biwi/biwi_eth.txt"
    other = "/x/OpenTraj/datasets/TrajNet/Train/crowds/students001.txt"
    assert bd._is_biwi_source(train) is True
    assert bd._is_biwi_source(test) is True
    assert bd._is_biwi_source(other) is False
    assert bd._source_role(train) == "raw_train_dir"
    assert bd._source_role(test) == "raw_test_dir"


def test_within_source_support_counts_are_agent_disjoint() -> None:
    source = "/x/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt"
    pool = {
        "source_file": np.asarray([source] * 8),
        "agent_id": np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64),
        "horizon": np.asarray([50, 10, 50, 10, 50, 10, 50, 10], dtype=np.int64),
    }
    out = bd._within_source_support_counts(pool, source)
    assert out["train_agent_count"] + out["val_agent_count"] == 4
    assert out["rows_by_split"]["train"] + out["rows_by_split"]["val"] == 8
    assert out["t50_rows_by_split"]["train"] + out["t50_rows_by_split"]["val"] == 4


def test_candidate_options_block_deployable_repair_when_only_support_is_current_test_source() -> None:
    hotel = "/x/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt"
    eth = "/x/OpenTraj/datasets/TrajNet/Test/biwi/biwi_eth.txt"
    pool = {
        "source_file": np.asarray([hotel, hotel, eth]),
        "agent_id": np.asarray([1, 2, 3], dtype=np.int64),
        "horizon": np.asarray([50, 10, 50], dtype=np.int64),
    }
    biwi_sources = {
        hotel: {"raw_role": "raw_train_dir", "rows": 2, "horizon_counts": {"50": 1}},
        eth: {"raw_role": "raw_test_dir", "rows": 1, "horizon_counts": {"50": 1}},
    }
    options = bd._candidate_options(pool, biwi_sources)
    assert options
    assert all(option["repair_training_allowed"] is False for option in options)
    assert any("no_independent_biwi_test_source_after_support_rebuild" in option["blockers"] for option in options)
    assert all(option.get("uses_raw_test_dir_for_training") is not True for option in options)


def test_gate_passes_for_conservative_preflight() -> None:
    payload = {
        "input_verdicts": {
            "stage43_bc": "stage43_bc_blocked_family_support_scan_pass",
            "stage43_f": "stage43_f_source_level_split_ready",
        },
        "summary": {
            "biwi_source_count_in_feature_store": 2,
            "current_train_rows": 0,
            "current_val_rows": 459,
            "current_test_rows": 7685,
            "candidate_option_count": 3,
            "deployable_repair_option_count": 0,
            "diagnostic_support_option_count": 2,
        },
        "candidate_rebuild_options": [
            {
                "repair_training_allowed": False,
                "uses_raw_test_dir_for_training": False,
            }
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
            "raw_test_dir_training": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "deployable_biwi_repair_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    gate = bd._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_bd_biwi_support_rebuild_preflight_pass"
