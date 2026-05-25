import numpy as np

from src import stage42_horizon_static_gate_repair as s42l


def test_stage42l_claim_boundary_blocks_overclaim() -> None:
    text = "\n".join(s42l.CURRENT_FACTS)
    assert "不是 true 3D" in text
    assert "raw-frame" in text
    assert "Stage5C" in text
    assert "SMC" in text


def test_stage42l_horizon_index_maps_known_horizons() -> None:
    idx = s42l._horizon_index_np(np.asarray([10, 25, 50, 100, 50]))
    assert idx.tolist() == [0, 1, 2, 3, 2]


def test_stage42l_gate_requires_t50_repair() -> None:
    base = {
        "rows": [{"seed": 83}, {"seed": 89}, {"seed": 97}],
        "summary": {
            "seeds": [83, 89, 97],
            "ade_all": {"mean": 0.02},
            "ade_t50": {"mean": 0.01},
            "ade_hard_failure": {"mean": 0.02},
            "ade_easy_degradation": {"mean": 0.0},
        },
        "comparison": {"stage42_k_fresh_static_gated": {"ade_t50": {"mean": -0.01}}},
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
    assert s42l._gate(base)["passed"] == s42l._gate(base)["total"]
    base["summary"]["ade_t50"]["mean"] = -0.001
    gate = s42l._gate(base)
    assert gate["passed"] < gate["total"]
    assert gate["gates"]["t50_ade_repaired"] is False
