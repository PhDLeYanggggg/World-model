import numpy as np

from src import stage41_pure_ucy_source_validation as pure_ucy


def test_subset_bundle_filters_row_aligned_arrays_recursively():
    data = {
        "features": np.arange(12).reshape(4, 3),
        "labels": {
            "horizon": np.array([10, 50, 100, 50]),
            "source_file": np.array(["a", "b", "c", "d"]),
            "scalar_note": "keep",
        },
        "scalar": 7,
    }
    subset = pure_ucy._subset_bundle(data, np.array([True, False, True, False]))
    assert subset["features"].shape == (2, 3)
    assert subset["labels"]["horizon"].tolist() == [10, 100]
    assert subset["labels"]["source_file"].tolist() == ["a", "c"]
    assert subset["labels"]["scalar_note"] == "keep"
    assert subset["scalar"] == 7


def test_strict_positive_requires_long_horizon_and_easy_safety():
    good = {
        "all_improvement": 0.01,
        "t50_improvement": 0.01,
        "t100_improvement": 0.01,
        "hard_failure_improvement": 0.01,
        "easy_degradation": 0.0,
        "collision_delta_vs_floor_005": 0.0,
    }
    assert pure_ucy._strict_positive(good)
    weak_t100 = dict(good, t100_improvement=0.0)
    assert not pure_ucy._strict_positive(weak_t100)
    unsafe_easy = dict(good, easy_degradation=0.03)
    assert not pure_ucy._strict_positive(unsafe_easy)


def test_target_source_locations_are_split_specific():
    val = {
        "labels": {
            "horizon": np.array([50, 50]),
            "source_file": np.array(
                [
                    "UCY/students01/students001-trajnet.txt",
                    "ETH/seq_eth/obsmat.txt",
                ]
            ),
        }
    }
    test = {
        "labels": {
            "horizon": np.array([50, 50]),
            "source_file": np.array(
                [
                    "UCY/zara03/crowds_zara03.txt",
                    "TrajNet/Train/crowds/crowds_zara03.txt",
                ]
            ),
        }
    }
    locations = pure_ucy._target_source_locations(val, test)
    assert locations["UCY__students01__students001-trajnet_txt"]["split"] == "val"
    assert locations["UCY__students01__students001-trajnet_txt"]["rows"] == 1
    assert locations["UCY__zara03__crowds_zara03_txt"]["split"] == "test"
    assert locations["UCY__zara03__crowds_zara03_txt"]["rows"] == 1
    assert all(not row["source"].startswith("TrajNet/") for row in locations.values())
