import numpy as np

from src import stage42_eth_ucy_source_specific_easy_guard as jg


def test_source_cv_split_uses_heldout_as_test_and_disjoint_sources():
    data = {
        "source_file": np.asarray(["a.txt", "a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]),
    }
    split, stats = jg._source_cv_split(data, "a.txt")
    assert set(split[data["source_file"] == "a.txt"]) == {"test"}
    assert stats["test_source"] == "a.txt"
    assert stats["source_overlap_pass"] is True
    assert stats["train_rows"] > 0
    assert stats["val_rows"] > 0


def test_subset_data_only_slices_row_aligned_arrays():
    data = {
        "x": np.arange(5),
        "matrix": np.arange(10).reshape(5, 2),
        "constant": np.arange(3),
    }
    out = jg._subset_data(data, np.asarray([True, False, True, False, True]))
    assert out["x"].tolist() == [0, 2, 4]
    assert out["matrix"].shape == (3, 2)
    assert out["constant"].tolist() == [0, 1, 2]


def test_summary_marks_partial_source_support():
    folds = [
        {
            "heldout_source": "good",
            "metrics": {
                "eth_ucy_source_specific_policy": {
                    "all_improvement": 0.1,
                    "t50_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.0,
                }
            },
        },
        {
            "heldout_source": "unsafe",
            "metrics": {
                "eth_ucy_source_specific_policy": {
                    "all_improvement": 0.1,
                    "t50_improvement": 0.1,
                    "hard_failure_improvement": 0.1,
                    "easy_degradation": 0.2,
                }
            },
        },
    ]
    summary = jg._summary(folds)
    assert summary["deployable_heldout_sources"] == ["good"]
    assert summary["blocked_heldout_sources"] == ["unsafe"]
    assert summary["decision"] == "eth_ucy_source_specific_policy_partial_source_support"
