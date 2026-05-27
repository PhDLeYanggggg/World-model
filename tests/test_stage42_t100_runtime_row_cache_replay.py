from __future__ import annotations

from pathlib import Path

import numpy as np

from src.stage42_group_consistency_t100_easy_guard_runtime import FrozenT100EasyGuardPolicy
from src.stage42_t100_runtime_row_cache_replay import (
    REQUIRED_CACHE_FIELDS,
    _cache_has_required_fields,
    _field_summary,
    _metric_from_errors,
    _replay_cache,
)


def _write_tiny_cache(path: Path) -> None:
    candidate = np.asarray(
        [
            [[10.0, 0.0], [20.0, 0.0]],
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 0.0], [4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    floor = np.zeros_like(candidate)
    selected = candidate.copy()
    selected[0] = floor[0]
    future = selected.copy()
    valid = np.ones((3, 2), dtype=bool)
    selected_ade = np.zeros(3, dtype=np.float32)
    selected_fde = np.zeros(3, dtype=np.float32)
    floor_ade = np.asarray([0.0, np.linalg.norm([2.0, 2.0]), 4.0], dtype=np.float32)
    floor_fde = floor_ade.copy()
    np.savez_compressed(
        path,
        row_id=np.arange(3, dtype=np.int64),
        split=np.asarray(["test", "test", "test"], dtype="U8"),
        domain=np.asarray(["TrajNet", "UCY", "TrajNet"], dtype="U64"),
        source_file=np.asarray(["a", "b", "c"], dtype="U16"),
        scene_id=np.asarray(["s", "s", "s"], dtype="U16"),
        frame_id=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        agent_id=np.asarray([1, 2, 3], dtype=np.int64),
        horizon=np.asarray([100, 100, 50], dtype=np.int64),
        candidate_xy_predicted_rollout=candidate,
        floor_xy_train_horizon_causal_rollout=floor,
        selected_xy_stage42_hr=selected,
        selected_xy_runtime_replay=selected,
        candidate_switch=np.asarray([True, True, True], dtype=bool),
        runtime_switch=np.asarray([False, True, True], dtype=bool),
        runtime_reason=np.asarray(
            [
                "validation_easy_harm_t100_fallback_floor",
                "validation_supported_t100_keep_candidate",
                "non_t100_not_guarded",
            ],
            dtype="U96",
        ),
        future_xy_label_eval_only=future,
        future_valid_label_eval_only=valid,
        normalizer=np.ones(3, dtype=np.float32),
        hard_label=np.asarray([False, True, False], dtype=bool),
        failure_label=np.asarray([False, False, True], dtype=bool),
        easy_label=np.asarray([True, False, False], dtype=bool),
        candidate_ade=np.zeros(3, dtype=np.float32),
        candidate_fde=np.zeros(3, dtype=np.float32),
        floor_ade=floor_ade,
        floor_fde=floor_fde,
        selected_ade=selected_ade,
        selected_fde=selected_fde,
    )


def _policy() -> FrozenT100EasyGuardPolicy:
    return FrozenT100EasyGuardPolicy(
        {
            "decision_table": {
                "guarded_slices": {"TrajNet|100": {"keep": False}},
                "kept_slices": {"UCY|100": {"keep": True}},
            },
            "decision_rule": {"threshold_easy_degradation": 0.0},
        },
        policy_hash="test",
    )


def test_required_cache_fields_are_present_in_tiny_cache(tmp_path: Path) -> None:
    path = tmp_path / "tiny.npz"
    _write_tiny_cache(path)
    summary = _field_summary(path)
    assert summary["exists"] is True
    assert summary["missing_required_fields"] == []
    assert set(REQUIRED_CACHE_FIELDS).issubset(set(summary["fields"]))


def test_cache_has_required_fields_false_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, row_id=np.arange(2))
    assert _cache_has_required_fields(path) is False
    summary = _field_summary(path)
    assert "candidate_xy_predicted_rollout" in summary["missing_required_fields"]


def test_runtime_replay_matches_stored_selected_xy(tmp_path: Path) -> None:
    path = tmp_path / "tiny.npz"
    _write_tiny_cache(path)
    replay = _replay_cache(_policy(), path)
    assert replay["rows"] == 3
    assert replay["selected_xy_max_abs_diff_vs_stored_hr"] == 0.0
    assert replay["switch_mismatch_vs_stored"] == 0
    assert replay["runtime_diagnostics"]["fallback_rows"] == 1
    assert replay["t100_rows"] == 2


def test_metric_from_errors_reports_easy_degradation() -> None:
    selected = np.asarray([1.0, 2.0, 3.0])
    floor = np.asarray([2.0, 2.0, 6.0])
    metric = _metric_from_errors(
        selected,
        floor,
        np.asarray([50, 100, 100]),
        np.asarray([False, True, False]),
        np.asarray([False, False, True]),
        np.asarray([True, False, False]),
        np.asarray([True, False, True]),
    )
    assert metric["all_improvement"] > 0
    assert metric["t100_raw_frame_diagnostic_improvement"] >= 0
    assert metric["hard_failure_improvement"] >= 0
    assert metric["easy_degradation"] < 0
