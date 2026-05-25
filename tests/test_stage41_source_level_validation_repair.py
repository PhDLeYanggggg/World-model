import numpy as np

from src import stage41_source_level_validation_repair as slv


def test_ucy_family_detector_covers_ucy_paths_and_crowds_sources():
    assert slv._is_ucy_family("/tmp/datasets/UCY/zara02/obsmat.txt", "ETH_UCY")
    assert slv._is_ucy_family("/tmp/datasets/TrajNet/Train/crowds/crowds_zara03.txt", "TrajNet")
    assert slv._is_ucy_family("/tmp/anything.txt", "UCY")
    assert not slv._is_ucy_family("/tmp/datasets/ETH/seq_eth/obsmat.txt", "ETH_UCY")


def test_source_stats_counts_horizons_and_slices():
    data = {
        "horizon": np.asarray([10, 50, 50, 100]),
        "scene_id": np.asarray(["a", "a", "b", "b"]),
        "source_file": np.asarray(["s1", "s1", "s2", "s2"]),
        "agent_id": np.asarray([1, 2, 2, 3]),
        "frame_id": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "hard": np.asarray([True, False, True, False]),
        "easy": np.asarray([False, True, False, True]),
        "failure": np.asarray([False, False, True, False]),
    }
    stats = slv._source_stats(data, np.asarray([True, True, False, False]))
    assert stats["rows"] == 2
    assert stats["sources"] == 1
    assert stats["scenes"] == 1
    assert stats["t50"] == 1
    assert stats["t100"] == 0
    assert stats["hard"] == 1
    assert stats["easy"] == 1


def test_positive_safe_requires_positive_core_and_easy_preserved():
    assert slv._positive_safe(
        {
            "all_improvement": 0.1,
            "t50_improvement": 0.0,
            "hard_failure_improvement": 0.2,
            "easy_degradation": 0.0,
        }
    )
    assert not slv._positive_safe(
        {
            "all_improvement": 0.1,
            "t50_improvement": 0.2,
            "hard_failure_improvement": 0.2,
            "easy_degradation": 0.03,
        }
    )
