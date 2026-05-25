import numpy as np

from src import stage42_retrained_ablation as s42g


def test_stage42g_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42g.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42g_select_from_pred_is_conservative() -> None:
    pred = np.array([[10.0, 9.0], [10.0, 1.0], [10.0, 9.9]], dtype=float)
    floor = np.array([0, 0, 0], dtype=int)
    selected = s42g._select_from_pred(pred, floor, confidence_min=0.2, gain_min=1.0, max_switch_rate=0.5)
    assert selected.tolist() == [0, 1, 0]


def test_stage42g_gate_requires_fresh_rows_and_boundaries() -> None:
    variants = [
        "full_retrained_external",
        "no_history",
        "no_neighbor",
        "no_goal",
        "no_scene_goal",
        "no_interaction",
        "no_domain_expert",
        "no_transformer_proxy_history_sequence",
        "no_safe_switch",
        "no_teacher_floor_proxy",
    ]
    summary = {
        name: {
            "seeds": [11, 17, 23],
            "all": {"mean": 0.1},
            "t50": {"mean": 0.1},
            "hard_failure": {"mean": 0.1},
            "easy_degradation": {"mean": 0.0},
        }
        for name in variants
    }
    contribution = {
        "no_history": {"t50_delta_full_minus_ablation": 0.01, "hard_delta_full_minus_ablation": 0.0},
        "no_neighbor": {"t50_delta_full_minus_ablation": 0.0, "hard_delta_full_minus_ablation": 0.02},
    }
    result = {
        "summary": summary,
        "contribution_vs_full": contribution,
        "rows": [{"source": "fresh_run"} for _ in range(30)],
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
    gate = s42g._gate(result)
    assert gate["passed"] == gate["total"]
