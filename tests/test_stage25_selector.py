from __future__ import annotations

import numpy as np

from src.stage25_pipeline import (
    BASELINE_NAMES,
    _evaluate_selection,
    _feature_names,
    _select_from_predicted_fde,
)


def _row(horizon: int = 50):
    return {
        "split_type": "cross_scene",
        "horizon": horizon,
        "target_agent_type": "Pedestrian",
        "agent_count": 6,
        "start_frame": 100,
        "hard_candidate": False,
        "goal_availability": "train_endpoint_or_visual_prior_by_split",
        "baseline_errors": {
            "constant_position": 20.0,
            "constant_velocity_causal_fd": 12.0,
            "damped_velocity": 10.0,
            "constant_acceleration_causal": 100.0,
            "constant_turn_rate_velocity": 12.0,
            "scene_clamped_baseline": 11.0,
            "goal_directed_baseline": 50.0,
        },
        "best_error": 10.0,
    }


def test_stage25_features_do_not_include_forbidden_oracle_inputs():
    joined = " ".join(_feature_names()).lower()
    assert "future" not in joined
    assert "endpoint" not in joined
    assert "oracle" not in joined
    assert "central_velocity" not in joined


def test_stage25_evaluate_strongest_selection_is_noop():
    rows = [_row()]
    metrics = {"strongest_baseline_by_split_horizon": {"cross_scene": {"50": {"baseline": "damped_velocity"}}}}
    out = _evaluate_selection(rows, ["damped_velocity"], metrics=metrics)
    assert out["official_t50_improvement"] == 0.0
    assert out["easy_degradation"] == 0.0
    assert out["harm_over_fallback"] == 0.0


def test_stage25_confidence_gate_falls_back_to_strongest():
    rows = [_row()]
    metrics = {"strongest_baseline_by_split_horizon": {"cross_scene": {"50": {"baseline": "damped_velocity"}}}}
    pred = np.ones((1, len(BASELINE_NAMES)), dtype=float) * 10.0
    pred[0, BASELINE_NAMES.index("constant_position")] = 9.5
    choice, conf = _select_from_predicted_fde(
        rows,
        pred,
        {"confidence_threshold": 0.5, "predicted_gain_threshold_px": 5.0, "easy_guard": True},
        metrics,
    )
    assert choice == ["damped_velocity"]
    assert conf[0] < 0.5
