import numpy as np

from src import stage42_eth_ucy_source_support_coverage as jl


def test_support_threshold_uses_pairwise_quantile():
    signatures = {
        "a": np.asarray([0.0, 0.0]),
        "b": np.asarray([3.0, 4.0]),
        "c": np.asarray([6.0, 8.0]),
    }
    threshold = jl._support_threshold(signatures)
    assert threshold > 0.0
    assert threshold <= 10.0


def test_nearest_sources_orders_by_distance():
    target = np.asarray([1.0, 0.0])
    signatures = {
        "near": np.asarray([1.1, 0.0]),
        "far": np.asarray([10.0, 0.0]),
    }
    nearest = jl._nearest_sources(target, signatures, k=2)
    assert [row["source"] for row in nearest] == ["near", "far"]


def test_is_family_safe_requires_easy_guard_and_positive_slice():
    assert jl._is_family_safe(
        {
            "all_improvement": 0.0,
            "t50_improvement": 0.04,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.01,
        }
    )
    assert not jl._is_family_safe(
        {
            "all_improvement": 0.10,
            "t50_improvement": 0.10,
            "hard_failure_improvement": 0.10,
            "easy_degradation": 0.03,
        }
    )
