import numpy as np

from src import stage41_domain_local_full_trajectory_world_state as dlft


def test_domain_summary_counts_waypoint_coverage() -> None:
    tensors = {
        "raw": {
            "horizon": np.asarray([10, 50, 100], dtype=np.int16),
            "hard": np.asarray([False, True, False]),
            "failure": np.asarray([False, False, True]),
            "easy": np.asarray([True, False, False]),
            "source_file": np.asarray(["a", "a", "b"]),
            "scene_id": np.asarray(["s1", "s1", "s2"]),
        },
        "traj": {
            "waypoint_valid": np.asarray(
                [
                    [True, True, True, True],
                    [False, False, False, True],
                    [True, False, True, True],
                ]
            )
        },
    }
    out = dlft._domain_summary(tensors)
    assert out["rows"] == 3
    assert out["full_waypoint_rows"] == 1
    assert out["t50_rows"] == 1
    assert out["t100_rows"] == 1
    assert out["sources"] == 2


def test_safe_metrics_empty_mask_returns_zeroes() -> None:
    labels = {
        "horizon": np.asarray([50]),
        "hard": np.asarray([False]),
        "failure": np.asarray([False]),
        "easy": np.asarray([True]),
        "domain": np.asarray(["unit"]),
        "candidate_fde": np.asarray([[1.0, 2.0]]),
    }
    out = dlft._safe_metrics(np.asarray([1.0]), np.asarray([1.0]), labels, np.asarray([False]), np.asarray([False]))
    assert out["rows"] == 0
    assert out["all_improvement"] == 0.0


def test_metric_ds_preserves_required_fields() -> None:
    labels = {
        "horizon": np.asarray([10, 50]),
        "hard": np.asarray([False, True]),
        "failure": np.asarray([False, False]),
        "easy": np.asarray([True, False]),
        "domain": np.asarray(["a", "b"]),
        "candidate_fde": np.ones((2, 3)),
        "ignored": np.asarray([0, 1]),
    }
    out = dlft._metric_ds(labels)
    assert sorted(out.keys()) == ["candidate_fde", "domain", "easy", "failure", "hard", "horizon"]
