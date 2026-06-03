from __future__ import annotations

import numpy as np

from src import stage43_t100_source_scene_supported_supervision_cache as cq


def test_assignment_by_agent_reconstructs_hash() -> None:
    pool = {
        "old_split": np.asarray(["old"] * 24),
        "local_row": np.arange(24),
        "dataset": np.asarray(["UCY"] * 24),
        "scene_id": np.asarray(["scene_a"] * 24),
        "source_file": np.asarray(["source_a"] * 24),
        "agent_id": np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6] * 2),
        "frame_id": np.arange(24),
        "horizon": np.asarray([100] * 24),
        "hard": np.zeros(24, dtype=bool),
        "failure": np.zeros(24, dtype=bool),
        "easy": np.ones(24, dtype=bool),
        "scale": np.ones(24),
    }
    assignments, assignment_hash = cq._assignment_by_agent_from_pool(pool)
    assert assignment_hash
    assert set(assignments.values()) == {"train", "val", "test"}
    assert assignments[cq._agent_key("source_a", "1")] in {"train", "val", "test"}


def test_gate_requires_t100_only_and_no_cache_commit() -> None:
    payload = {
        "stage43_cp_precondition": {"verdict": "stage43_cp_t100_source_scene_support_split_ready"},
        "split_summaries": {
            "train": {
                "cache_path": "README_RESULTS.md",
                "rows": 10,
                "horizon_counts": {"100": 10},
                "full_waypoint_rows": 10,
                "max_endpoint_diff_last_waypoint": 0.0,
            },
            "val": {
                "cache_path": "README_RESULTS.md",
                "rows": 5,
                "horizon_counts": {"100": 5},
                "full_waypoint_rows": 5,
                "max_endpoint_diff_last_waypoint": 0.0,
            },
            "test": {
                "cache_path": "README_RESULTS.md",
                "rows": 5,
                "horizon_counts": {"100": 5},
                "full_waypoint_rows": 5,
                "max_endpoint_diff_last_waypoint": 0.0,
            },
        },
        "support_summary": {"source_or_scene_supported_ratio": 1.0},
        "no_leakage": {
            "source_agent_disjoint": True,
            "row_disjoint": True,
            "source_scene_overlap_intentional_for_support_protocol": True,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "cache_committed": False,
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }
    gate = cq._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["t100_supported_supervised_training_ready"] is True


def test_overlap_counts_supports_source_agent_key() -> None:
    split_arrays = {
        "train": {"source_file": np.asarray(["a"]), "agent_id": np.asarray([1]), "old_split": np.asarray(["train"]), "local_row": np.asarray([0]), "scene_id": np.asarray(["s"])},
        "val": {"source_file": np.asarray(["a"]), "agent_id": np.asarray([2]), "old_split": np.asarray(["train"]), "local_row": np.asarray([1]), "scene_id": np.asarray(["s"])},
        "test": {"source_file": np.asarray(["a"]), "agent_id": np.asarray([3]), "old_split": np.asarray(["val"]), "local_row": np.asarray([0]), "scene_id": np.asarray(["s"])},
    }
    assert cq._overlap_counts(split_arrays, "source_agent") == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert cq._overlap_counts(split_arrays, "row_key") == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert cq._overlap_counts(split_arrays, "source_file") == {"train_val": 1, "train_test": 1, "val_test": 1}
