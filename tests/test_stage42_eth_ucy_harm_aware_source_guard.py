import numpy as np

from src import stage42_eth_ucy_harm_aware_source_guard as jh


def test_cap_by_score_keeps_top_scores_per_horizon():
    base = np.asarray([True, True, True, True, True, True])
    score = np.asarray([0.1, 0.9, 0.5, 0.2, 0.8, 0.3])
    horizon = np.asarray([10, 10, 10, 50, 50, 50])
    capped = jh._cap_by_score(base, score, horizon, 0.34)
    assert capped.tolist() == [False, True, False, False, True, False]


def test_thresholds_from_validation_includes_zero_and_quantiles():
    score = np.asarray([-1.0, 0.5, 2.0, 4.0])
    mask = np.asarray([True, True, True, False])
    thresholds = jh._thresholds_from_validation(score, mask)
    assert 0.0 in thresholds
    assert min(thresholds) == -1.0
    assert max(thresholds) == 2.0


def test_summary_records_easy_repaired_sources():
    folds = [
        {
            "heldout_source": "repaired",
            "metrics": {
                "harm_aware_source_guard": {
                    "all_improvement": 0.1,
                    "t50_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                },
                "base_horizon_policy_before_harm_guard": {"easy_degradation": 0.2},
            },
        },
        {
            "heldout_source": "blocked",
            "metrics": {
                "harm_aware_source_guard": {
                    "all_improvement": 0.1,
                    "t50_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.3,
                },
                "base_horizon_policy_before_harm_guard": {"easy_degradation": 0.4},
            },
        },
    ]
    summary = jh._summary(folds)
    assert summary["deployable_heldout_sources"] == ["repaired"]
    assert summary["blocked_heldout_sources"] == ["blocked"]
    assert summary["easy_repaired_sources"] == ["repaired"]
