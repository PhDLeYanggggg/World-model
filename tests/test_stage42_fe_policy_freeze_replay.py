import numpy as np

from src import stage42_fe_policy_freeze_replay as ff


def test_stable_hash_is_order_invariant() -> None:
    a = {"mode": "fc_to_safety", "threshold": 0.05, "nested": {"b": 2, "a": 1}}
    b = {"nested": {"a": 1, "b": 2}, "threshold": 0.05, "mode": "fc_to_safety"}

    assert ff._stable_hash(a) == ff._stable_hash(b)
    assert len(ff._stable_hash(a)) == 64


def test_bootstrap_rate_delta_detects_negative_delta() -> None:
    a = np.asarray([0, 0, 1, 0, 0, 0], dtype=bool)
    b = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)

    ci = ff._bootstrap_rate_delta(a, b, seed=7, n=100)

    assert ci["bootstrap_n"] == 0  # small synthetic arrays are intentionally not overclaimed
    large_a = np.resize(a, 60)
    large_b = np.resize(b, 60)
    large_ci = ff._bootstrap_rate_delta(large_a, large_b, seed=7, n=100)
    assert large_ci["bootstrap_n"] == 100
    assert large_ci["mid"] < 0.0


def test_gate_blocks_metric_seconds_stage5c_and_smc() -> None:
    payload = {
        "source": ff.SOURCE,
        "fe_artifact": {"exists": True},
        "frozen_policy": {"policy_hash": "a" * 64},
        "replay": {
            "candidate_exact_replay": True,
            "max_metric_abs_diff": 0.0,
            "max_diagnostic_abs_diff": 0.0,
            "test_metric_vs_floor": {
                "all_improvement": 0.1,
                "t50_improvement": 0.1,
                "hard_failure_improvement": 0.1,
                "easy_degradation": 0.0,
            },
            "near_delta_vs_fc": -0.1,
            "near_delta_vs_di": 0.0,
        },
        "bootstrap_ci": {
            "all": {"low": 0.1, "high": 0.2, "bootstrap_n": 2000},
            "t50": {"low": 0.1, "high": 0.2, "bootstrap_n": 2000},
            "t100_raw_frame_diagnostic": {"low": 0.1, "high": 0.2, "bootstrap_n": 2000},
            "hard_failure": {"low": 0.1, "high": 0.2, "bootstrap_n": 2000},
            "easy_degradation": {"low": -0.1, "high": 0.0, "bootstrap_n": 2000},
        },
        "near_bootstrap_ci": {
            "final_near_005": {"bootstrap_n": 2000},
            "delta_final_minus_fc": {"bootstrap_n": 2000},
            "delta_final_minus_di": {"bootstrap_n": 2000},
            "delta_final_minus_fb": {"bootstrap_n": 2000},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = ff._gate(payload)

    assert gate["verdict"] == "stage42_ff_fe_policy_freeze_replay_pass"
    assert gate["gates"]["no_metric_seconds_overclaim"] is True
    assert gate["gates"]["stage5c_false"] is True
    assert gate["gates"]["smc_false"] is True
