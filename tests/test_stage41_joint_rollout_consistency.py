import numpy as np

from src import stage41_joint_rollout_consistency as jrc


def _labels():
    return {
        "current_xy": np.asarray([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]], dtype=float),
        "normalizer": np.ones(3, dtype=float),
        "horizon": np.asarray([50, 50, 100]),
        "hard": np.asarray([True, False, True]),
        "failure": np.asarray([False, False, True]),
        "easy": np.asarray([False, True, False]),
        "domain": np.asarray(["D", "D", "E"]),
        "candidate_fde": np.asarray([[1.0, 0.5], [1.0, 1.5], [1.0, 0.25]], dtype=float),
    }


def test_group_counts_align_to_rows():
    counts = jrc._group_counts(np.asarray(["a", "a", "b", "c", "c", "c"], dtype=object))
    assert counts.tolist() == [2, 2, 1, 3, 3, 3]


def test_subset_metrics_respects_mask_and_easy_guard():
    labels = _labels()
    selected = np.asarray([0.5, 1.1, 0.25], dtype=float)
    floor = np.asarray([1.0, 1.0, 1.0], dtype=float)
    switch = np.asarray([True, True, True])
    metrics = jrc._subset_metrics(selected, floor, labels, switch, np.asarray([True, True, False]))
    assert metrics["rows"] == 2
    assert metrics["t50_improvement"] > 0.0
    assert metrics["easy_degradation"] > 0.0


def test_joint_stats_reports_mixed_switch_and_collision_rates():
    labels = _labels()
    xy = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.01, 0.0], [0.01, 0.0], [0.01, 0.0], [0.01, 0.0]],
            [[4.0, 0.0], [4.1, 0.0], [4.2, 0.0], [4.3, 0.0]],
        ],
        dtype=float,
    )
    keys = np.asarray(["g", "g", "h"], dtype=object)
    switch = np.asarray([True, False, False])
    stats = jrc._joint_stats("toy", xy, labels, keys, switch)
    assert stats["multi_agent_rows"] == 2
    assert stats["mixed_group_switch_rate"] == 0.5
    assert stats["near_collision_rate_005"] > 0.0


def test_proximity_guard_turns_off_unsafe_switch_only():
    labels = _labels()
    floor_xy = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[4.0, 0.0], [4.1, 0.0], [4.2, 0.0], [4.3, 0.0]],
        ],
        dtype=float,
    )
    neural_xy = floor_xy.copy()
    neural_xy[0, :, 0] = 0.99
    neural_xy[0, :, 1] = 0.0
    keys = np.asarray(["g", "g", "h"], dtype=object)
    switch = np.asarray([True, False, True])
    guarded, guarded_off = jrc._apply_proximity_guard(floor_xy, neural_xy, labels, keys, switch, min_sep=0.05)
    assert guarded_off == 1
    assert guarded.tolist() == [False, False, True]
