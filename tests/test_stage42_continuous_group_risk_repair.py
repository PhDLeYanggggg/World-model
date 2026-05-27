from __future__ import annotations

import numpy as np

from src import stage42_continuous_group_risk_repair as ey


def test_group_reduce_max_assigns_group_max_to_all_rows() -> None:
    values = np.asarray([0.1, 0.7, 0.3, 0.2, 0.9])
    group_key = np.asarray(["g1", "g1", "g2", "g3", "g3"], dtype=object)
    out = ey._group_reduce_max(values, group_key)
    assert np.allclose(out, [0.7, 0.7, 0.3, 0.9, 0.9])


def test_bucket_thresholds_use_unique_group_scores() -> None:
    scores = np.asarray([0.1, 0.1, 0.5, 0.9, 0.9])
    group_key = np.asarray(["g1", "g1", "g2", "g3", "g3"], dtype=object)
    ids = np.arange(5)
    thresholds = ey._bucket_thresholds(scores, group_key, ids, 3)
    assert len(thresholds) == 2
    buckets = ey._bucketize(scores, thresholds)
    assert buckets[0] == buckets[1]
    assert buckets[3] == buckets[4]
    assert len(set(buckets.tolist())) >= 2


def test_bucket_key_modes() -> None:
    data = {
        "dataset": np.asarray(["UCY", "TrajNet", "UCY"], dtype=object),
        "horizon": np.asarray([50, 100, 50], dtype=np.int64),
    }
    ids = np.asarray([0, 1, 2], dtype=np.int64)
    buckets = np.asarray([0, 2, 1], dtype=np.int64)
    assert ey._bucket_key(data, ids, buckets, "global").tolist() == ["global", "global", "global"]
    assert ey._bucket_key(data, ids, buckets, "domain_horizon").tolist() == ["UCY|50", "TrajNet|100", "UCY|50"]
    assert ey._bucket_key(data, ids, buckets, "domain_horizon_risk3").tolist() == ["UCY|50|risk0", "TrajNet|100|risk2", "UCY|50|risk1"]


def test_deployment_decision_requires_di_lift() -> None:
    metric = {"all_improvement": 0.2, "hard_failure_improvement": 0.2, "easy_degradation": 0.0}
    selected = {
        "test_diagnostics": {"final_near_005": 0.01, "base_near_005": 0.02},
        "mixed_group_selection": {"mixed_group_count": 0},
    }
    ok = ey._deployment_decision(metric, {"delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.02}}, selected)
    not_ok = ey._deployment_decision(metric, {"delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": -0.02}}, selected)
    assert ok["promote_continuous_group_risk_repair"] is True
    assert not_ok["promote_continuous_group_risk_repair"] is False


def test_gate_promotable_payload() -> None:
    payload = {
        "continuous_group_risk_repair": {
            "candidate_count": 84,
            "risk_score_stats": {"val_unique_group_scores": 5},
            "mode_rows": [
                {"mode": "global", "bucket_counts_val": {"0": 3}},
                {"mode": "domain_horizon", "bucket_counts_val": {"0": 3}},
                {"mode": "domain_horizon_risk3", "bucket_counts_val": {"0": 1, "1": 1, "2": 1}},
                {"mode": "domain_horizon_risk4", "bucket_counts_val": {"0": 1, "1": 1, "2": 1, "3": 1}},
            ],
            "selected": {
                "mode": "domain_horizon_risk3",
                "test_metric_vs_floor": {
                    "all_improvement": 0.25,
                    "t50_improvement": 0.2,
                    "hard_failure_improvement": 0.24,
                    "easy_degradation": 0.0,
                },
                "test_diagnostics": {"final_near_005": 0.01, "base_near_005": 0.02},
            },
        },
        "comparison_to_prior": {"delta_vs_stage42_di": {"all_improvement": 0.01, "hard_failure_improvement": 0.01}},
        "deployment_decision": {"group_consistent_selection": True},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_bucket_thresholds": True,
            "validation_only_rule_selection": True,
            "validation_only_mode_selection": True,
            "source_overlap_pass": True,
        },
        "claim_boundary": {
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = ey._gate(payload)
    assert gate["passed"] == gate["total"]
