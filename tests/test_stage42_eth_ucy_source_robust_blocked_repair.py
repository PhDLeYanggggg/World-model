import numpy as np

from src import stage42_eth_ucy_source_robust_blocked_repair as ji


def test_candidate_score_rejects_easy_unsafe_support():
    metric = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.1,
    }
    support = {
        "mean_all_improvement": 0.1,
        "mean_t50_improvement": 0.1,
        "max_easy_degradation": 0.03,
        "min_all_improvement": 0.1,
    }
    assert ji._candidate_score(metric, support) < -1e8


def test_policy_errors_applies_horizon_alpha():
    pred = np.asarray([[[2.0, 0.0]], [[0.0, 4.0]]], dtype=np.float32)
    floor = np.asarray([[[0.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32)
    labels = {
        "waypoint_xy": np.asarray([[[1.0, 0.0]], [[0.0, 2.0]]], dtype=np.float32),
        "waypoint_valid": np.ones((2, 1), dtype=bool),
    }
    policy = {"slices": {"h10": {"alpha": 0.5}}}
    ade, fde, switch = ji._policy_errors(pred, floor, labels, policy, np.asarray([10, 50]))
    assert np.allclose(ade, [0.0, 2.0])
    assert np.allclose(fde, [0.0, 2.0])
    assert switch.tolist() == [True, False]


def test_deployable_rule_requires_positive_and_easy_safe():
    assert ji._test_deployable(
        {
            "all_improvement": 0.05,
            "t50_improvement": 0.04,
            "hard_failure_improvement": 0.02,
            "easy_degradation": 0.01,
        }
    )
    assert not ji._test_deployable(
        {
            "all_improvement": 0.05,
            "t50_improvement": 0.04,
            "hard_failure_improvement": 0.02,
            "easy_degradation": 0.03,
        }
    )
