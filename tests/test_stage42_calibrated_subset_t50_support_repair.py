import numpy as np

from src import stage42_calibrated_subset_t50_support_repair as bq


def test_family_source_counts_unique_sources() -> None:
    rel = np.asarray(
        [
            "ETH/seq_eth/obsmat.txt",
            "ETH/seq_eth/obsmat.txt",
            "ETH/seq_hotel/obsmat.txt",
            "UCY/zara01/obsmat.txt",
            "UCY/zara02/obsmat.txt",
            "UCY/students03/obsmat.txt",
        ]
    )
    counts = bq._family_source_counts(rel, np.ones(len(rel), dtype=bool))
    assert counts == {"ETH_seq": 2, "UCY_zara": 2, "UCY_students": 1}


def test_t50_requires_two_family_sources() -> None:
    counts = {"ETH_seq": 1, "UCY_zara": 2, "UCY_students": 1}
    assert bq._eligible_families_for_horizon(counts, 50) == {"UCY_zara"}
    assert bq._eligible_families_for_horizon(counts, 25) == {"ETH_seq", "UCY_zara", "UCY_students"}


def test_gate_requires_t50_nonnegative_and_blocks_metric_claim() -> None:
    payload = {
        "source": "fresh_calibrated_subset_t50_support_repair",
        "bp_verdict": "stage42_bp_calibrated_subset_safety_repair_pass_limited_positive",
        "summary": {
            "source_cv_folds": 6,
            "calibrated_sources_evaluated": 6,
            "nonnegative_all_folds": True,
            "nonnegative_t50_folds": True,
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
    gate = bq._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["limited_positive_t50_nonharm"] is True
    assert gate["verdict"] == "stage42_bq_calibrated_subset_t50_support_repair_pass_t50_nonharm_limited_positive"
