from __future__ import annotations

import numpy as np

from src import stage42_fh_horizon_row_switch_specialist as fm


def _eval(ade: list[float], xy_shift: float = 0.0, switch: bool = True) -> dict:
    n = len(ade)
    xy = np.zeros((n, 4, 2), dtype=np.float32)
    xy[:, :, 0] = xy_shift
    return {
        "selected_xy": xy,
        "selected_ade": np.asarray(ade, dtype=np.float64),
        "selected_fde": np.asarray(ade, dtype=np.float64),
        "floor_ade": np.ones(n, dtype=np.float64) * 10.0,
        "floor_fde": np.ones(n, dtype=np.float64) * 10.0,
        "switch": np.ones(n, dtype=bool) if switch else np.zeros(n, dtype=bool),
    }


def test_apply_rule_to_arrays_switches_only_matching_rows() -> None:
    base = _eval([5.0, 5.0, 5.0], xy_shift=0.0)
    cand = _eval([4.0, 8.0, 3.0], xy_shift=1.0)
    feature = np.asarray([0.1, 0.5, 0.9])
    local = np.asarray([True, False, True])

    out = fm._apply_rule_to_arrays(base, cand, feature, local, "ge", 0.5)

    assert np.allclose(out["selected_ade"], [5.0, 5.0, 3.0])
    assert out["use"].tolist() == [False, False, True]


def test_score_penalizes_easy_and_near_excess() -> None:
    good = {
        "all_improvement": 0.2,
        "hard_failure_improvement": 0.2,
        "t50_improvement": 0.1,
        "t100_raw_frame_diagnostic_improvement": 0.1,
        "easy_degradation": 0.0,
        "switch_rate": 0.1,
    }
    bad = dict(good)
    bad["easy_degradation"] = 0.5

    assert fm._score(good, near_proxy=0.0, base_near_proxy=0.0) > fm._score(bad, near_proxy=0.5, base_near_proxy=0.0)


def test_gate_keeps_horizon_limit_when_weak_slices_remain() -> None:
    payload = {
        "source": fm.SOURCE,
        "summary": {
            "fl_root_cause_counts": {"oracle_label_low_margin_ambiguous": 3},
            "weak_horizon_count_before": 3,
            "weak_horizon_count_after": 3,
            "applied_policies": {"UCY|50": {"mode": "feature_threshold", "switch_rows": 10}},
        },
        "policies": {"UCY|50": {"selected": {"policy": {"mode": "feature_threshold"}}}},
        "selection_rule": {"uses_test_metrics_for_policy_selection": False},
        "metric_vs_floor": {
            "all_improvement": 0.3,
            "t50_improvement": 0.2,
            "hard_failure_improvement": 0.2,
            "easy_degradation": 0.0,
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
            "uniform_horizon_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = fm._gate(payload)

    assert gate["verdict"] == "stage42_fm_horizon_row_switch_specialist_pass_with_horizon_limit"
