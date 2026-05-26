import numpy as np

from src import stage42_source_level_safety_floor_audit as s42at


def test_feature_masks_separate_floor_and_family_context():
    names = [
        "safe_baseline_rel_0",
        "family_baseline_rel_0",
        "floor_rel_x",
        "horizon_50",
        "domain_UCY",
        "history_scalar_0",
    ]
    masks = s42at._feature_masks(names)
    assert masks["baseline_family_all_context"].tolist() == [True, True, True, True, True, False]
    assert masks["no_floor_rel_context"].tolist() == [True, True, False, True, True, False]
    assert masks["family_only_no_floor_safe_context"].tolist() == [False, True, False, True, True, False]
    assert masks["no_safe_baseline_context"].tolist() == [False, True, True, True, True, False]


def test_deployable_requires_positive_and_easy_safe():
    ok = {"all_improvement": 0.1, "t50_improvement": 0.0, "hard_failure_improvement": 0.2, "easy_degradation": 0.01}
    bad_easy = dict(ok, easy_degradation=0.03)
    bad_gain = dict(ok, all_improvement=0.0)
    assert s42at._deployable(ok)
    assert not s42at._deployable(bad_easy)
    assert not s42at._deployable(bad_gain)


def test_slice_safety_reports_val_and_test_rows():
    n = 80
    data = {
        "dataset": np.asarray(["A"] * n),
        "horizon": np.asarray([50] * n),
        "hard": np.asarray([i % 2 == 0 for i in range(n)]),
        "failure": np.asarray([i % 3 == 0 for i in range(n)]),
        "easy": np.asarray([i % 2 == 1 for i in range(n)]),
    }
    split = np.asarray(["val"] * 40 + ["test"] * 40)
    selected = np.asarray([1.0] * 40 + [0.8] * 40)
    floor = np.ones(n)
    out = s42at._slice_safety(data, split, selected, floor)
    assert "A|50" in out
    assert out["A|50"]["val_rows"] == 40
    assert out["A|50"]["test_rows"] == 40


def test_gate_accepts_fallback_partial_removal_without_overclaim():
    metric = {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0}
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "candidates": {
            "baseline_family_all_context": {
                "protected_safe_switch": metric,
                "ungated_all_rows": metric,
                "bootstrap": {"ungated_all_rows": {"all": {"bootstrap_n": 1000}, "t50": {"bootstrap_n": 1000}}},
            }
        },
        "context_deltas": {"a": {}, "b": {}, "c": {}},
        "summary": {"teacher_floor_context_removal": "not_supported_as_global_replacement"},
        "slice_safety_for_all_context_ungated": {"A|50": {}},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42at._gate(result)
    assert gate["passed"] == gate["total"]
