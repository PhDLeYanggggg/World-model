import numpy as np

from src import stage42_source_rotation_easy_guard_repair as jf


def test_cap_switch_mask_limits_switches_by_horizon():
    base = np.asarray([True] * 10 + [True] * 10)
    residual = np.linspace(0.0, 1.0, 20)
    horizon = np.asarray([10] * 10 + [50] * 10)
    policy = {
        "slices": {
            "h10": {"direction": "high"},
            "h50": {"direction": "low"},
        }
    }
    capped = jf._cap_switch_mask(base, residual, policy, horizon, 0.2)
    assert int(np.sum(capped & (horizon == 10))) == 2
    assert int(np.sum(capped & (horizon == 50))) == 2
    assert set(np.where(capped & (horizon == 10))[0]) == {8, 9}
    assert set(np.where(capped & (horizon == 50))[0]) == {10, 11}


def test_candidate_score_rejects_easy_harm():
    assert jf._candidate_score(
        {
            "all_improvement": 1.0,
            "t50_improvement": 1.0,
            "hard_failure_improvement": 1.0,
            "easy_degradation": 0.03,
            "switch_rate": 0.1,
        }
    ) < -1e8


def test_summary_marks_still_easy_blocked_domain():
    rotations = [
        {
            "heldout_domain": "ETH_UCY",
            "metrics": {
                "easy_guard_policy": {
                    "all_improvement": 0.2,
                    "t50_improvement": 0.2,
                    "hard_failure_improvement": 0.2,
                    "easy_degradation": 0.3,
                },
                "base_horizon_policy_before_cap": {"easy_degradation": 0.4},
            },
        },
        {
            "heldout_domain": "UCY",
            "metrics": {
                "easy_guard_policy": {
                    "all_improvement": 0.2,
                    "t50_improvement": 0.2,
                    "hard_failure_improvement": 0.2,
                    "easy_degradation": 0.0,
                },
                "base_horizon_policy_before_cap": {"easy_degradation": 0.0},
            },
        },
    ]
    summary = jf._summary(rotations)
    assert summary["still_easy_blocked_domains"] == ["ETH_UCY"]
    assert summary["deployable_heldout_domains_after_easy_guard"] == ["UCY"]
