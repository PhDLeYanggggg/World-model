import numpy as np

from src import stage42_aw_t100_easy_safety_repair as s42ay


def test_t100_slice_keep_uses_strict_easy_nonharm():
    keep = {"val_metric": {"all_improvement": 0.1, "easy_degradation": 0.0}}
    unsafe_easy = {"val_metric": {"all_improvement": 0.1, "easy_degradation": 0.001}}
    nonpositive = {"val_metric": {"all_improvement": 0.0, "easy_degradation": -0.1}}
    assert s42ay._t100_slice_keep(keep)
    assert not s42ay._t100_slice_keep(unsafe_easy)
    assert not s42ay._t100_slice_keep(nonpositive)


def test_apply_t100_easy_guard_falls_back_unsafe_slice():
    data = {
        "dataset": np.asarray(["TrajNet", "TrajNet", "UCY"]),
        "horizon": np.asarray([100, 50, 100]),
    }
    policy = {
        "slices": {
            "TrajNet|100": {"val_metric": {"all_improvement": 0.1, "easy_degradation": 0.02}},
            "UCY|100": {"val_metric": {"all_improvement": 0.1, "easy_degradation": -0.01}},
        }
    }
    selected = np.asarray([5.0, 4.0, 3.0])
    floor = np.asarray([10.0, 8.0, 6.0])
    switch = np.asarray([True, True, True])
    out = s42ay._apply_t100_easy_guard(
        policy=policy,
        data=data,
        selected_ade=selected,
        selected_fde=selected.copy(),
        switch=switch,
        floor_ade=floor,
        floor_fde=floor.copy(),
    )
    assert out["selected_ade"][0] == floor[0]
    assert out["switch"][0] is np.False_
    assert out["selected_ade"][1] == selected[1]
    assert out["selected_ade"][2] == selected[2]
    assert "TrajNet|100" in out["guarded_slices"]
    assert "UCY|100" in out["kept_slices"]


def test_domain_positive_requires_gain_and_easy_safety():
    ok = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
    }
    assert s42ay._domain_positive(ok)
    assert not s42ay._domain_positive(dict(ok, t50_improvement=0.0))
    assert not s42ay._domain_positive(dict(ok, easy_degradation=0.03))


def test_gate_passes_t100_easy_repair():
    metric = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "t100_raw_frame_diagnostic_improvement": 0.1,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
    }
    result = {
        "source": "fresh_run",
        "aw_verdict": "stage42_aw_ucy_validation_support_repair_pass",
        "ax_verdict": "stage42_ax_repaired_protocol_robustness_pass_with_t100_limit",
        "repair_policy": {"uses_test_metrics_for_threshold": False},
        "repair_effect": {"guarded_slice_count": 1},
        "repaired_metrics": {
            "by_horizon": {"100": metric},
            "by_domain": {"TrajNet": metric, "UCY": metric},
            "bootstrap": {
                "h100_easy_degradation": {"high": 0.01},
                "all": {"low": 0.01},
                "t50": {"low": 0.01},
                "t100_raw_frame_diagnostic": {"low": 0.01},
                "hard_failure": {"low": 0.01},
                "easy_degradation": {"high": 0.01},
            },
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "t100_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42ay._gate(result)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ay_t100_easy_safety_repair_pass"
