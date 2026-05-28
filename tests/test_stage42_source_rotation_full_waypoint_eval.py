import numpy as np

from src import stage42_source_rotation_full_waypoint_eval as je


def test_leave_one_domain_split_keeps_heldout_as_test_and_disjoint_sources():
    data = {
        "dataset": np.asarray(["A", "A", "B", "B", "C", "C"]),
        "source_file": np.asarray(["a1", "a2", "b1", "b2", "c1", "c2"]),
    }
    split, stats = je._leave_one_domain_split(data, "C")
    assert set(split[data["dataset"] == "C"]) == {"test"}
    assert stats["test_domains"] == ["C"]
    assert stats["source_overlap_pass"] is True
    assert stats["train_rows"] + stats["val_rows"] + stats["test_rows"] == 6


def test_summary_records_positive_and_deployable_domains():
    rotations = [
        {
            "heldout_domain": "A",
            "metrics": {
                "protected_horizon_policy": {
                    "all_improvement": 0.05,
                    "t50_improvement": 0.04,
                    "hard_failure_improvement": 0.02,
                    "easy_degradation": 0.0,
                }
            },
        },
        {
            "heldout_domain": "B",
            "metrics": {
                "protected_horizon_policy": {
                    "all_improvement": -0.01,
                    "t50_improvement": 0.0,
                    "hard_failure_improvement": 0.0,
                    "easy_degradation": 0.0,
                }
            },
        },
    ]
    summary = je._summary(rotations)
    assert summary["positive_heldout_domains"] == ["A"]
    assert summary["deployable_heldout_domains"] == ["A"]
    assert summary["decision"] == "source_rotation_positive_but_not_global_deployable"


def test_gate_passes_completed_audit_even_when_rotation_is_negative():
    payload = {
        "domains": ["A", "B", "C"],
        "rotations": [
            {
                "split_stats": {"test_rows": 10, "train_rows": 20, "val_rows": 5, "source_overlap_pass": True},
                "feature_schema": {"domain_features_removed": ["domain_A"]},
                "policy": {"domain_specific_thresholds": False},
                "bootstrap": {"all": {"bootstrap_n": 1000}},
            }
            for _ in range(3)
        ],
        "summary": {"positive_heldout_domain_count": 0, "all_rotations_no_easy_harm": False},
        "no_leakage": {
            "source_overlap_pass": True,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "domain_specific_test_thresholds": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = je._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_je_source_rotation_full_waypoint_eval_pass"
