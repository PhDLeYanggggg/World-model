import numpy as np

from src import stage42_ucy_zara_t50_family_policy as bs


def test_zara_folds_are_source_disjoint() -> None:
    rel = np.asarray(
        [
            "UCY/zara01/obsmat.txt",
            "UCY/zara02/obsmat.txt",
            "UCY/zara03/crowds_zara03.txt",
        ]
    )
    for fold in bs._build_zara_folds():
        split = bs._split_for_fold(rel, fold)
        assert set(split.tolist()) == {"train", "val", "test"}
        assert not (set(rel[split == "train"]) & set(rel[split == "val"]))
        assert not (set(rel[split == "train"]) & set(rel[split == "test"]))
        assert not (set(rel[split == "val"]) & set(rel[split == "test"]))


def test_gate_passes_honest_blocker_without_positive_claim() -> None:
    payload = {
        "source": "fresh_ucy_zara_t50_family_policy",
        "br_verdict": "stage42_br_calibrated_t50_source_support_gap_audit_pass",
        "summary": {
            "zara_sources_present": 3,
            "source_cv_folds": 3,
            "t50_rows_total": 100,
            "candidate_t50_oracle_headroom_macro_mean": 0.1,
            "easy_degradation_max": 0.0,
            "positive_t50_claim_allowed": False,
            "positive_t50_fold_count": 0,
            "nonnegative_all_folds": True,
            "easy_safe_all_folds": True,
            "t50_improvement_macro_mean": 0.0,
        },
        "no_leakage": {
            "source_overlap_pass": True,
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
    gate = bs._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["positive_t50_transfer"] is False
    assert gate["verdict"] == "stage42_bs_ucy_zara_t50_family_policy_pass_honest_blocker"


def test_gate_marks_positive_only_when_all_t50_folds_positive_and_safe() -> None:
    payload = {
        "source": "fresh_ucy_zara_t50_family_policy",
        "br_verdict": "stage42_br_calibrated_t50_source_support_gap_audit_pass",
        "summary": {
            "zara_sources_present": 3,
            "source_cv_folds": 3,
            "t50_rows_total": 100,
            "candidate_t50_oracle_headroom_macro_mean": 0.1,
            "easy_degradation_max": 0.0,
            "positive_t50_claim_allowed": True,
            "positive_t50_fold_count": 3,
            "nonnegative_all_folds": True,
            "easy_safe_all_folds": True,
            "t50_improvement_macro_mean": 0.05,
        },
        "no_leakage": {
            "source_overlap_pass": True,
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
    gate = bs._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["positive_t50_transfer"] is True
    assert gate["verdict"] == "stage42_bs_ucy_zara_t50_family_policy_pass_positive"
