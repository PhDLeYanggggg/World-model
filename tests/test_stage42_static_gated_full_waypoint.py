import numpy as np

from src import stage42_static_gated_full_waypoint as s42j


def test_stage42j_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42j.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42j_mix_prediction_interpolates() -> None:
    a = {"waypoint_delta": np.zeros((2, 4, 2), dtype=np.float32), "traj_risk": np.zeros(2, dtype=np.float32)}
    b = {"waypoint_delta": np.ones((2, 4, 2), dtype=np.float32), "traj_risk": np.ones(2, dtype=np.float32)}
    mixed = s42j._mix_pred(a, b, 0.25)
    assert np.allclose(mixed["waypoint_delta"], 0.25)
    assert np.allclose(mixed["traj_risk"], 0.25)


def test_stage42j_gate_accepts_static_gated_repair() -> None:
    metric = {
        "source": "cached_verified_checkpoints_fresh_static_gate_eval",
        "seeds": [53, 59, 61],
        "ade_all": {"mean": 0.02},
        "ade_t50": {"mean": 0.03},
        "ade_t100_raw_frame_diagnostic": {"mean": 0.01},
        "ade_hard_failure": {"mean": 0.02},
        "ade_easy_degradation": {"mean": 0.0},
        "fde_all": {"mean": 0.02},
        "fde_t50": {"mean": 0.04},
        "switch_rate": {"mean": 0.05},
    }
    result = {
        "checkpoint_status": "available",
        "summary": {
            "static_gated": metric,
            "full_static": {**metric, "ade_all": {"mean": -0.01}, "ade_t50": {"mean": -0.02}},
            "no_static": {**metric, "ade_all": {"mean": 0.02}, "ade_t50": {"mean": 0.03}},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42j._gate(result)
    assert gate["passed"] == gate["total"]
