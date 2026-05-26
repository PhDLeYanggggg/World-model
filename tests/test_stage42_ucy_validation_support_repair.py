import numpy as np

from src import stage42_ucy_validation_support_repair as s42aw


def test_ucy_internal_val_group_selects_smallest_train_group():
    split = np.asarray(["train", "train", "train", "test"])
    group = np.asarray(["big", "small", "big", "test"])
    domain = np.asarray(["UCY", "UCY", "UCY", "UCY"])
    assert s42aw._ucy_internal_val_group(split, group, domain) == "small"


def test_split_with_ucy_internal_val_does_not_touch_test():
    split = np.asarray(["train", "train", "test", "val"], dtype="U8")
    group = np.asarray(["a", "b", "c", "d"])
    domain = np.asarray(["UCY", "UCY", "UCY", "TrajNet"])
    repaired, val_group = s42aw._split_with_ucy_internal_val(split, group, domain)
    assert val_group in {"a", "b"}
    assert repaired[2] == "test"
    assert np.sum(repaired == "val") == 2


def test_positive_ucy_requires_gain_easy_safety_and_switch():
    ok = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.5,
    }
    assert s42aw._positive_ucy(ok)
    assert not s42aw._positive_ucy(dict(ok, switch_rate=0.0))
    assert not s42aw._positive_ucy(dict(ok, easy_degradation=0.03))
    assert not s42aw._positive_ucy(dict(ok, t50_improvement=0.0))


def test_gate_accepts_ucy_repair():
    metric = {
        "all_improvement": 0.2,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.2,
        "easy_degradation": 0.0,
        "switch_rate": 0.5,
    }
    result = {
        "summary": {"ucy_val_rows_after": 10},
        "no_leakage": {
            "internal_val_from_train_only": True,
            "test_sources_unchanged": True,
            "source_overlap_pass": True,
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "variants": {"a": {}, "b": {}, "c": {}, "d": {}},
        "validation_best": {
            "by_domain": {"UCY": metric, "TrajNet": metric},
            "protected": metric,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42aw._gate(result)
    assert gate["passed"] == gate["total"]
