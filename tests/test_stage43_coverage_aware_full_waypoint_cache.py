from __future__ import annotations

import numpy as np

from src import stage43_coverage_aware_full_waypoint_cache as m


def test_concat_for_split_uses_ce_assignments(monkeypatch) -> None:
    geo = {
        "source_file": np.asarray(["s1", "s2", "s1"]),
        "dataset": np.asarray(["D", "D", "D"]),
        "scene_id": np.asarray(["a", "b", "a"]),
        "agent_id": np.asarray([1, 2, 1]),
        "frame_id": np.asarray([0.0, 1.0, 2.0]),
        "current_x": np.asarray([0.0, 1.0, 2.0]),
        "current_y": np.asarray([0.0, 1.0, 2.0]),
        "future_endpoint_x": np.asarray([1.0, 2.0, 3.0]),
        "future_endpoint_y": np.asarray([1.0, 2.0, 3.0]),
        "horizon": np.asarray([50, 50, 100]),
        "dt_frame_step": np.asarray([1.0, 1.0, 1.0]),
        "track_length": np.asarray([10.0, 10.0, 10.0]),
        "valid_mask": np.asarray([True, True, True]),
    }
    labels = {
        "scale": np.asarray([1.0, 1.0, 1.0]),
        "hard": np.asarray([True, False, True]),
        "failure": np.asarray([False, False, True]),
        "easy": np.asarray([False, True, False]),
    }

    def fake_load(split: str):
        if split == "train":
            return geo, labels
        empty_geo = {key: value[:0] for key, value in geo.items()}
        empty_labels = {key: value[:0] for key, value in labels.items()}
        return empty_geo, empty_labels

    monkeypatch.setattr(m, "_load_old_split", fake_load)
    out = m._concat_for_split({"s1": "test", "s2": "val"}, "test")
    assert out["source_file"].tolist() == ["s1", "s1"]
    assert out["local_row"].tolist() == [0, 2]
    assert out["old_split"].tolist() == ["train", "train"]


def test_source_sets_are_disjoint() -> None:
    arrays = {
        "train": {"source_file": np.asarray(["a", "b"])},
        "val": {"source_file": np.asarray(["c"])},
        "test": {"source_file": np.asarray(["d"])},
    }
    sets = m._source_sets(arrays)
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])


def test_gate_accepts_ready_cache(tmp_path) -> None:
    files = {}
    for split in ["train", "val", "test"]:
        path = tmp_path / f"{split}.npz"
        path.write_bytes(b"cache")
        files[split] = str(path)
    summary = {
        split: {
            "cache_path": files[split],
            "rows": 10,
            "full_waypoint_rows": 10,
            "max_endpoint_diff_last_waypoint": 0.0,
        }
        for split in ["train", "val", "test"]
    }
    payload = {
        "stage43_ce_precondition": {
            "verdict": "stage43_ce_source_family_coverage_split_repair_ready",
            "split_rows": {"train": 10, "val": 10, "test": 10},
        },
        "split_summaries": summary,
        "source_overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
        "coverage_summary": {
            "test_families_without_validation_support": [],
            "test_domain_families_without_validation_support": [],
        },
        "no_leakage": {
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "cache_committed": False,
        "claim_boundary": {
            "new_training_or_evaluation_not_run": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    gate = m._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cf_coverage_aware_full_waypoint_cache_ready"
