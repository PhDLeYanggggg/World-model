import numpy as np

from src import stage42_ay_shadow_holdout_robustness as s42az


def test_val_t100_source_count_counts_unique_shadow_val_groups():
    data = {
        "dataset": np.asarray(["A", "A", "A", "B"]),
        "horizon": np.asarray([100, 100, 50, 100]),
    }
    split = np.asarray(["shadow_val", "shadow_val", "shadow_val", "shadow_val"])
    group = np.asarray(["g1", "g2", "g1", "g3"])
    assert s42az._val_t100_source_count(data, split, group, "A") == 2
    assert s42az._val_t100_source_count(data, split, group, "B") == 1


def test_source_support_guard_requires_multiple_validation_sources():
    data = {
        "dataset": np.asarray(["ETH_UCY", "ETH_UCY", "ETH_UCY"]),
        "horizon": np.asarray([100, 100, 50]),
    }
    split = np.asarray(["shadow_val", "shadow_holdout", "shadow_holdout"])
    group = np.asarray(["g_val", "g_hold", "g_hold"])
    policy = {
        "slices": {
            "ETH_UCY|100": {
                "val_metric": {"all_improvement": 0.2, "easy_degradation": -0.1}
            }
        }
    }
    selected = np.asarray([1.0, 2.0, 3.0])
    floor = np.asarray([10.0, 20.0, 30.0])
    switch = np.asarray([True, True, True])
    out = s42az._apply_source_support_t100_guard(
        policy=policy,
        data=data,
        group=group,
        split=split,
        selected_ade=selected,
        selected_fde=selected.copy(),
        switch=switch,
        floor_ade=floor,
        floor_fde=floor.copy(),
        min_sources=2,
    )
    assert "ETH_UCY|100" in out["guarded_slices"]
    assert out["selected_ade"][1] == floor[1]
    assert out["selected_ade"][2] == selected[2]


def test_source_support_guard_keeps_when_support_and_easy_safe():
    data = {
        "dataset": np.asarray(["ETH_UCY", "ETH_UCY", "ETH_UCY"]),
        "horizon": np.asarray([100, 100, 100]),
    }
    split = np.asarray(["shadow_val", "shadow_val", "shadow_holdout"])
    group = np.asarray(["g1", "g2", "g3"])
    policy = {
        "slices": {
            "ETH_UCY|100": {
                "val_metric": {"all_improvement": 0.2, "easy_degradation": -0.1}
            }
        }
    }
    selected = np.asarray([1.0, 2.0, 3.0])
    floor = np.asarray([10.0, 20.0, 30.0])
    switch = np.asarray([True, True, True])
    out = s42az._apply_source_support_t100_guard(
        policy=policy,
        data=data,
        group=group,
        split=split,
        selected_ade=selected,
        selected_fde=selected.copy(),
        switch=switch,
        floor_ade=floor,
        floor_fde=floor.copy(),
        min_sources=2,
    )
    assert "ETH_UCY|100" in out["kept_slices"]
    assert out["selected_ade"][2] == selected[2]


def test_gate_passes_with_ay_limitation_and_safe_conservative_guard():
    metric = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "t100_raw_frame_diagnostic_improvement": 0.0,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
    }
    result = {
        "source": "fresh_run",
        "ay_verdict": "stage42_ay_t100_easy_safety_repair_pass",
        "shadow_split_stats": {"shadow_holdout": {"rows": 10}},
        "summary": {
            "ay_strict_guard_shadow_h100_easy_safe": False,
            "source_support_guard_t100_positive": False,
            "ucy_shadow_status": "not_run",
        },
        "policies": {"source_support_t100_guard": {"guarded_slices": {"ETH_UCY|100": {}}}},
        "shadow_holdout_metrics": {
            "source_support_t100_guard": {
                "protected": metric,
                "by_horizon": {"100": metric},
                "bootstrap": {
                    "h100_easy_degradation": {"high": 0.01},
                    "all": {"low": 0.01},
                    "t50": {"low": 0.01},
                    "hard_failure": {"low": 0.01},
                },
            }
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "final_test_metrics_for_threshold": False,
            "shadow_fit_val_holdout_from_original_train_only": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "t100_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42az._gate(result)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_az_shadow_holdout_robustness_pass_with_ay_t100_limitation"
