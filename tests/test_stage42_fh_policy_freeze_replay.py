from __future__ import annotations

import numpy as np

from src import stage42_fh_policy_freeze_replay as fi


def _metric(all_i: float = 0.2, t50: float = 0.1, hard: float = 0.2, easy: float = 0.0) -> dict:
    return {
        "rows": 10,
        "all_improvement": all_i,
        "t50_improvement": t50,
        "t100_raw_frame_diagnostic_improvement": 0.1,
        "hard_failure_improvement": hard,
        "easy_degradation": easy,
    }


def _payload() -> dict:
    return {
        "source": fi.SOURCE,
        "fh_artifact": {"exists": True},
        "frozen_policy": {"policy_hash": "b" * 64},
        "replay": {
            "candidate_exact_replay": True,
            "max_metric_abs_diff": 0.0,
            "max_diagnostic_abs_diff": 0.0,
            "test_metric_vs_floor": _metric(),
            "by_domain": {"UCY": _metric(), "TrajNet": _metric()},
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
            "internal_val_from_train_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_positive_safe_requires_motion_gain_and_easy_safety() -> None:
    assert fi._positive_safe(_metric())
    assert not fi._positive_safe(_metric(all_i=0.0))
    assert not fi._positive_safe(_metric(t50=0.0))
    assert not fi._positive_safe(_metric(hard=0.0))
    assert not fi._positive_safe(_metric(easy=0.03))


def test_gate_passes_frozen_fh_policy_with_dual_domain_support() -> None:
    gate = fi._gate(_payload())

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_fi_fh_policy_freeze_replay_pass"
    assert gate["gates"]["ucy_positive_safe"] is True
    assert gate["gates"]["trajnet_positive_safe"] is True


def test_gate_fails_when_ucy_replay_is_not_positive_safe() -> None:
    payload = _payload()
    payload["replay"]["by_domain"]["UCY"] = _metric(all_i=0.0, t50=0.0, hard=0.0)

    gate = fi._gate(payload)

    assert gate["passed"] < gate["total"]
    assert gate["gates"]["ucy_positive_safe"] is False


def test_by_domain_metric_helper_is_deterministic() -> None:
    data = {
        "dataset": np.asarray(["UCY", "UCY", "TrajNet"]),
        "horizon": np.asarray([50, 50, 100]),
        "hard": np.asarray([True, False, True]),
        "failure": np.asarray([False, False, True]),
        "easy": np.asarray([False, True, False]),
    }
    ids = np.asarray([0, 1, 2])
    selected = np.asarray([1.0, 1.0, 1.0])
    floor = np.asarray([2.0, 2.0, 2.0])
    switch = np.asarray([True, True, True])

    out = fi._by_domain(data, ids, selected, floor, switch)

    assert sorted(out) == ["TrajNet", "UCY"]
    assert out["UCY"]["rows"] == 2
