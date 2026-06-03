from __future__ import annotations

import numpy as np

from src import stage43_t100_source_scene_support_split_repair as cp


def test_agent_disjoint_source_supported_assignment_has_all_splits() -> None:
    pool = {
        "old_split": np.asarray(["old"] * 24),
        "local_row": np.arange(24),
        "dataset": np.asarray(["UCY"] * 24),
        "scene_id": np.asarray(["scene_a"] * 24),
        "source_file": np.asarray(["source_a"] * 12 + ["source_b"] * 12),
        "agent_id": np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6] * 2),
        "frame_id": np.arange(24),
        "horizon": np.asarray([10, 100] * 12),
        "hard": np.zeros(24, dtype=bool),
        "failure": np.zeros(24, dtype=bool),
        "easy": np.ones(24, dtype=bool),
        "scale": np.ones(24),
    }
    assignments, plan = cp._assign_agent_disjoint_source_supported(pool)
    assert set(assignments.tolist()) == {"train", "val", "test"}
    assert not plan["blockers"]["sources_with_too_few_agents"]
    leakage = cp._leakage_summary(pool, assignments)
    assert leakage["row_disjoint"] is True
    assert leakage["source_agent_disjoint"] is True
    assert any(value > 0 for value in leakage["source_file_overlap_counts"].values())


def test_support_summary_counts_exact_source_scene_support() -> None:
    pool = {
        "source_file": np.asarray(["a"] * 6 + ["b"] * 6),
        "scene_id": np.asarray(["s1"] * 6 + ["s2"] * 6),
        "horizon": np.asarray([100] * 12),
    }
    assignments = np.asarray(["train", "val", "val", "test", "test", "test"] * 2)
    support = cp._support_summary(pool, assignments, min_support_rows=2)
    assert support["test_t100_rows"] == 6
    assert support["source_supported_test_t100_rows"] == 6
    assert support["scene_supported_test_t100_rows"] == 6
    assert support["exact_source_scene_supported_test_t100_rows"] == 6
    assert support["source_or_scene_supported_ratio"] == 1.0


def test_gate_records_not_model_result_boundary() -> None:
    payload = {
        "stage43_f_precondition": {"verdict": "stage43_f_source_level_split_ready"},
        "assignment_hash": "abc",
        "split_summary": {
            "train": {"rows": 10},
            "val": {"rows": 5},
            "test": {"rows": 5},
        },
        "support_summary": {
            "test_t100_rows": 3,
            "source_or_scene_supported_ratio": 1.0,
            "exact_source_scene_supported_ratio": 1.0,
        },
        "no_leakage": {
            "row_disjoint": True,
            "source_agent_disjoint": True,
            "source_scene_overlap_intentional_for_support_protocol": True,
            "cross_source_generalization_split": False,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "new_training_or_evaluation_not_run": True,
            "requires_cache_rebuild_before_training": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    gate = cp._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cp_t100_source_scene_support_split_ready"
