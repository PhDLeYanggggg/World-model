import numpy as np

from src import stage42_calibrated_subset_eval as bo


def test_source_cv_folds_hold_out_each_source_once() -> None:
    source_ids = ["A", "B", "C", "D"]
    folds = bo._build_source_cv_folds(source_ids)
    assert [fold["holdout_source"] for fold in folds] == source_ids
    assert all(fold["holdout_source"] not in fold["train_sources"] for fold in folds)
    assert all(fold["validation_source"] not in fold["train_sources"] for fold in folds)


def test_split_for_fold_has_no_source_overlap() -> None:
    rel = np.asarray(["a.txt", "b.txt", "c.txt", "d.txt", "other.txt"])
    original = bo.SOURCE_TO_REL.copy()
    try:
        bo.SOURCE_TO_REL.clear()
        bo.SOURCE_TO_REL.update({"A": "a.txt", "B": "b.txt", "C": "c.txt", "D": "d.txt"})
        fold = {"train_sources": ["A", "B"], "validation_source": "C", "test_sources": ["D"]}
        split = bo._split_for_fold(rel, fold)
        assert set(rel[split == "train"]) == {"a.txt", "b.txt"}
        assert set(rel[split == "val"]) == {"c.txt"}
        assert set(rel[split == "test"]) == {"d.txt"}
        assert split[-1] == "ignore"
    finally:
        bo.SOURCE_TO_REL.clear()
        bo.SOURCE_TO_REL.update(original)


def test_gate_blocks_global_metric_even_for_positive_subset() -> None:
    payload = {
        "source": "fresh_calibrated_subset_source_cv",
        "bn_verdict": "stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked",
        "summary": {
            "calibrated_sources_evaluated": 6,
            "source_cv_folds": 6,
            "source_overlap_pass": True,
            "positive_all_folds": True,
            "positive_t50_folds": True,
            "easy_degradation_max": 0.01,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = bo._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["positive_transfer"] is True
    assert gate["verdict"] == "stage42_bo_calibrated_subset_eval_pass_positive_claim_limited"
