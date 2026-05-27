from __future__ import annotations

import numpy as np

from src import stage42_fh_horizon_conservative_easy_guard as fn


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


def test_apply_guard_replaces_only_threshold_rows() -> None:
    base = _eval([5.0, 5.0, 5.0], xy_shift=0.0)
    replacement = _eval([10.0, 4.0, 3.0], xy_shift=1.0, switch=False)
    feature = np.asarray([0.1, 0.5, 0.9])
    local = np.asarray([True, True, False])

    out, use = fn._apply_guard(base, replacement, feature, local, "ge", 0.5)

    assert out["selected_ade"].tolist() == [5.0, 4.0, 5.0]
    assert use.tolist() == [False, True, False]
    assert out["switch"].tolist() == [True, False, True]


def test_score_penalizes_easy_ci_and_near_ci() -> None:
    row = {
        "metric": {
            "all_improvement": 0.2,
            "hard_failure_improvement": 0.2,
            "t50_improvement": 0.1,
            "t100_raw_frame_diagnostic_improvement": 0.1,
            "switch_rate": 0.2,
        },
        "bootstrap": {"easy_degradation": {"high": 0.0}},
        "near_bootstrap": {"delta_final_minus_fc": {"high": 0.0}},
    }
    bad = {
        "metric": dict(row["metric"]),
        "bootstrap": {"easy_degradation": {"high": 0.2}},
        "near_bootstrap": {"delta_final_minus_fc": {"high": 0.2}},
    }

    assert fn._score(row) > fn._score(bad)


def test_gate_blocks_uniform_claim_when_weak_horizons_remain() -> None:
    payload = {
        "source": fn.SOURCE,
        "summary": {
            "fm_verdict": "stage42_fm_horizon_row_switch_specialist_pass_with_horizon_limit",
            "weak_horizon_count_before": 2,
            "weak_horizon_count_after": 1,
            "applied_guards": {"UCY|100": {"mode": "feature_guard", "guard_rows": 12}},
        },
        "policies": {"UCY|100": {"selected": {"policy": {"mode": "feature_guard"}}}},
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

    gate = fn._gate(payload)

    assert gate["verdict"] == "stage42_fn_conservative_easy_guard_pass_with_horizon_limit"
