import numpy as np

from src import stage41_policy_blender as blender


def test_stage41_policy_blender_prefers_safe_positive_slice() -> None:
    good = {"improvement": 0.04, "hard_failure_improvement": 0.02, "easy_degradation": 0.0, "switch_rate": 0.1}
    bad = {"improvement": 0.04, "hard_failure_improvement": 0.02, "easy_degradation": 0.05, "switch_rate": 0.1}
    assert blender._score_for_mode(good, 50, "long_horizon") > blender._score_for_mode(bad, 50, "long_horizon")


def test_stage41_policy_blender_slice_metrics_easy_degradation() -> None:
    selected = np.array([0.8, 1.2, 1.0])
    fallback = np.ones(3)
    switch = np.array([True, True, False])
    ds = {
        "hard": np.array([False, True, False]),
        "failure": np.array([False, False, False]),
        "easy": np.array([True, False, False]),
    }
    out = blender._slice_metrics(selected, fallback, ds, switch, np.ones(3, dtype=bool))
    assert out["improvement"] == 0.0
    assert out["easy_degradation"] == 0.0
    assert out["switch_rate"] == 2 / 3
