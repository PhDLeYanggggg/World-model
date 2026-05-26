import numpy as np

from src import stage42_calibrated_subset_safety_repair as bp


def test_source_guard_rejects_easy_harm() -> None:
    selected = np.asarray([0.8, 2.0, 0.8, 0.8])
    floor = np.asarray([1.0, 1.0, 1.0, 1.0])
    switch = np.asarray([True, True, True, True])
    data = {
        "horizon": np.asarray([50, 50, 50, 50]),
        "hard": np.asarray([False, False, True, True]),
        "failure": np.asarray([False, False, False, False]),
        "easy": np.asarray([True, True, False, False]),
    }
    support = np.asarray([True, True, True, True])
    rel_source = np.asarray(["a", "a", "a", "a"])
    assert bp._source_guard_ok(selected, floor, data, switch, support, rel_source) is False


def test_source_guard_accepts_safe_positive_support() -> None:
    selected = np.asarray([0.99, 0.99, 0.7, 0.7, 0.98, 0.98])
    floor = np.ones_like(selected)
    switch = np.asarray([False, False, True, True, True, True])
    data = {
        "horizon": np.asarray([50, 50, 50, 50, 100, 100]),
        "hard": np.asarray([False, False, True, True, True, True]),
        "failure": np.asarray([False, False, False, False, False, False]),
        "easy": np.asarray([True, True, False, False, False, False]),
    }
    support = np.asarray([True, True, True, True, True, True])
    rel_source = np.asarray(["a", "a", "a", "a", "a", "a"])
    assert bp._source_guard_ok(selected, floor, data, switch, support, rel_source) is True


def test_source_family_separates_ucy_students_from_zara() -> None:
    rel = np.asarray(
        [
            "UCY/students03/obsmat.txt",
            "UCY/zara01/obsmat.txt",
            "UCY/zara02/obsmat.txt",
            "ETH/seq_eth/obsmat.txt",
        ]
    )
    assert bp._source_family(rel).tolist() == ["UCY_students", "UCY_zara", "UCY_zara", "ETH_seq"]


def test_gate_blocks_global_metric_claim_and_tracks_limited_positive() -> None:
    payload = {
        "source": "fresh_calibrated_subset_safety_repair",
        "bo_verdict": "stage42_bo_calibrated_subset_eval_partial",
        "summary": {
            "source_cv_folds": 6,
            "calibrated_sources_evaluated": 6,
            "nonnegative_all_folds": True,
            "easy_safe_all_folds": True,
            "positive_fold_count": 2,
            "positive_t50_fold_count": 1,
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
    gate = bp._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["limited_positive_transfer"] is True
    assert gate["verdict"] == "stage42_bp_calibrated_subset_safety_repair_pass_limited_positive"
