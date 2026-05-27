from __future__ import annotations

import numpy as np

from src import stage42_fh_horizon_gain_harm_specialist as fo


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


def test_ridge_fit_predicts_linear_gain() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    coef = fo._ridge_fit(x, y, alpha=1e-6)
    pred = fo._ridge_predict(x, coef)
    assert np.allclose(pred, y, atol=1e-3)


def test_apply_predictions_respects_gain_harm_and_switch_cap() -> None:
    base = _eval([5.0, 5.0, 5.0, 5.0], xy_shift=0.0)
    cand = _eval([4.0, 4.0, 8.0, 3.0], xy_shift=1.0, switch=False)
    evals = {"di": cand}
    local = np.asarray([True, True, True, True])
    pred = {
        "best_gain": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "best_harm": np.asarray([0.0, 0.1, 0.9, 0.0], dtype=np.float32),
        "best_candidate": np.asarray(["di", "di", "di", "di"], dtype=object),
    }
    out, use = fo._apply_predictions(base, evals, local, pred, gain_min=0.0, harm_max=0.2, max_switch=0.5)

    assert int(np.sum(use)) == 2
    assert out["selected_ade"].tolist() == [4.0, 4.0, 5.0, 5.0]


def test_gate_keeps_horizon_limit_when_weak_horizons_remain() -> None:
    payload = {
        "source": fo.SOURCE,
        "summary": {
            "fn_verdict": "stage42_fn_conservative_easy_guard_pass_with_horizon_limit",
            "weak_horizon_count_before": 2,
            "weak_horizon_count_after": 2,
            "applied_policies": {"TrajNet|100": {"mode": "gain_harm_model", "switch_rows": 10}},
        },
        "model_summaries": {"TrajNet|100": {"feature_dim": 10}},
        "selection_rule": {
            "uses_test_metrics_for_policy_selection": False,
            "uses_future_labels_for_training_only": True,
        },
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

    gate = fo._gate(payload)

    assert gate["verdict"] == "stage42_fo_gain_harm_specialist_pass_with_horizon_limit"
