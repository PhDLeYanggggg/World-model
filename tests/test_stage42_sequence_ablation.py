import numpy as np

from src import stage42_sequence_ablation as s42h


def test_stage42h_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42h.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42h_horizon_one_hot_shape() -> None:
    x, names = s42h._horizon_one_hot(np.asarray([10, 25, 50, 100, 25]))
    assert x.shape == (5, 4)
    assert names == ["horizon_10", "horizon_25", "horizon_50", "horizon_100"]
    assert x[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert x[-1].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_stage42h_gate_accepts_sequence_evidence() -> None:
    variants = [
        "sequence_full_safe_switch",
        "sequence_no_history_tokens",
        "sequence_no_goal_scene_tokens",
        "sequence_no_neighbor_interaction_tokens",
        "sequence_full_no_safe_switch",
    ]
    summary = {
        name: {
            "seeds": [31, 37, 43],
            "all": {"mean": 0.1},
            "t50": {"mean": 0.1},
            "hard_failure": {"mean": 0.1},
            "easy_degradation": {"mean": 0.0},
        }
        for name in variants
    }
    result = {
        "summary": summary,
        "contribution_vs_sequence_full": {
            "sequence_no_history_tokens": {"t50_delta_full_minus_ablation": 0.01, "hard_delta_full_minus_ablation": 0.0},
            "sequence_no_goal_scene_tokens": {"t50_delta_full_minus_ablation": 0.0, "hard_delta_full_minus_ablation": 0.02},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42h._gate(result)
    assert gate["passed"] == gate["total"]
