import numpy as np

from src import stage42_context_slice_policy_promotion as ja


def test_rule_supported_requires_rows_lift_and_easy_safety():
    baseline = {
        "all_improvement": 0.10,
        "t50_improvement": 0.10,
        "t100_raw_frame_diagnostic_improvement": 0.0,
        "hard_failure_improvement": 0.10,
    }
    good = {
        "all_improvement": 0.12,
        "t50_improvement": 0.10,
        "t100_raw_frame_diagnostic_improvement": 0.0,
        "hard_failure_improvement": 0.10,
        "easy_degradation": 0.0,
    }
    unsafe = dict(good, easy_degradation=0.05)
    assert ja._rule_supported(good, baseline, ja.MIN_VAL_SLICE_ROWS)
    assert not ja._rule_supported(good, baseline, ja.MIN_VAL_SLICE_ROWS - 1)
    assert not ja._rule_supported(unsafe, baseline, ja.MIN_VAL_SLICE_ROWS)


def test_compose_policy_applies_validation_rules_without_touching_other_rows():
    n = 4
    data = {
        "dataset": np.asarray(["A", "A", "B", "B"]),
        "horizon": np.asarray([50, 50, 25, 25]),
        "hard": np.asarray([True, False, True, False]),
        "failure": np.asarray([False, False, False, False]),
        "easy": np.asarray([False, True, False, True]),
    }
    shared = {"split": np.asarray(["test"] * n), "data": data}
    outputs = {
        "tree_baseline_family_residual": {
            "selected_ade": np.asarray([8.0, 8.0, 8.0, 8.0]),
            "selected_fde": np.asarray([9.0, 9.0, 9.0, 9.0]),
            "switch": np.asarray([False, False, False, False]),
            "floor_ade": np.asarray([10.0, 10.0, 10.0, 10.0]),
            "floor_fde": np.asarray([10.0, 10.0, 10.0, 10.0]),
        },
        "tree_full_residual": {
            "selected_ade": np.asarray([6.0, 6.0, 7.0, 7.0]),
            "selected_fde": np.asarray([7.0, 7.0, 8.0, 8.0]),
            "switch": np.asarray([True, True, True, True]),
            "floor_ade": np.asarray([10.0, 10.0, 10.0, 10.0]),
            "floor_fde": np.asarray([10.0, 10.0, 10.0, 10.0]),
        },
    }
    slices = {
        "slice_a": np.asarray([True, True, False, False]),
        "slice_b": np.asarray([False, False, True, True]),
    }
    rules = [{"slice": "slice_a", "trial": "tree_full_residual", "val_score": 1.0}]
    policy = ja._compose_policy(shared, outputs, slices, rules)
    assert policy["test_rows_covered_by_context_rule"] == 2
    assert policy["owner_counts_test"]["tree_full_residual"] == 2
    assert policy["owner_counts_test"]["tree_baseline_family_residual"] == 2
    assert policy["metrics"]["context_slice_policy"]["all_improvement"] > policy["metrics"]["baseline_family_reference"]["all_improvement"]


def test_gate_records_not_promotable_when_no_rules():
    result = {
        "split_stats": {"by_split": {"test": {"rows": 47458}}},
        "validation_rule_diagnostics": {
            "selection_source": "validation_only",
            "test_threshold_tuning": False,
            "selected_rule_count": 0,
            "inference_safe_slice_filter": True,
        },
        "selected_rules": [],
        "summary": {
            "metrics": {"context_slice_policy": {"rows": 47458, "easy_degradation": 0.0}},
            "delta_vs_baseline_family": {
                "all_improvement": 0.0,
                "t50_improvement": 0.0,
                "t100_raw_frame_diagnostic_improvement": 0.0,
                "hard_failure_improvement": 0.0,
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_slice_thresholds": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = ja._gate(result)
    assert not gate["gates"]["selected_context_rules_present"]
    assert gate["verdict"] == "stage42_ja_context_slice_policy_not_promotable"


def test_validation_rules_skip_label_only_audit_slices():
    shared = {
        "split": np.asarray(["val", "val", "val", "val"]),
        "data": {
            "dataset": np.asarray(["A", "A", "A", "A"]),
            "horizon": np.asarray([50, 50, 50, 50]),
            "hard": np.asarray([True, False, True, False]),
            "failure": np.asarray([False, False, False, False]),
            "easy": np.asarray([False, True, False, True]),
        },
    }
    outputs = {
        "tree_baseline_family_residual": {
            "selected_ade": np.asarray([8.0, 8.0, 8.0, 8.0]),
            "floor_ade": np.asarray([10.0, 10.0, 10.0, 10.0]),
            "switch": np.asarray([False, False, False, False]),
        },
        "tree_full_residual": {
            "selected_ade": np.asarray([6.0, 6.0, 6.0, 6.0]),
            "floor_ade": np.asarray([10.0, 10.0, 10.0, 10.0]),
            "switch": np.asarray([True, True, True, True]),
        },
    }
    slices = {
        "all_test": np.asarray([True, True, True, True]),
        "hard_failure": np.asarray([True, False, True, False]),
        "easy": np.asarray([False, True, False, True]),
        "horizon:50": np.asarray([True, True, True, True]),
    }
    rules, diagnostics = ja._validation_rules(shared, outputs, slices)
    assert diagnostics["inference_safe_slice_filter"]
    assert all(r["slice"] not in ja.DISALLOWED_POLICY_SLICES for r in rules)


def test_stage42_ja_run_completes_or_uses_cache():
    result = ja.run_stage42_context_slice_policy_promotion(use_cached=True)
    if not result:
        result = ja.run_stage42_context_slice_policy_promotion()
    assert result["stage42_ja_gate"]["gates"]["validation_only_rule_selection"]
    assert result["stage42_ja_gate"]["gates"]["test_once_policy_evaluated"]
    assert result["no_leakage"]["future_waypoint_input"] is False
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False
