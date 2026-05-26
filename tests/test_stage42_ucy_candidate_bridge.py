import numpy as np

from src import stage42_ucy_candidate_bridge as u


def test_linear_endpoint_waypoints_uses_stage42_waypoint_fractions():
    current = np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    endpoint = np.asarray([[10.0, 20.0], [3.0, 3.0]], dtype=np.float64)
    out = u._linear_endpoint_waypoints(current, endpoint)
    assert out.shape == (2, 4, 2)
    np.testing.assert_allclose(out[:, 0, :], current + 0.25 * (endpoint - current))
    np.testing.assert_allclose(out[:, -1, :], endpoint)


def test_gate_marks_failed_bridge_when_ucy_metric_negative():
    result = {
        "stage41_checkpoint_exists": True,
        "no_leakage": {"future_endpoint_input": False},
        "claim_boundary": {"stage5c_executed": False, "smc_enabled": False},
        "val_bridge": {"matched_rows": 10},
        "test_bridge": {
            "metrics": {
                "UCY_zara03_test": {
                    "matched_rows": 10,
                    "switch_rate": 0.5,
                    "ade": {"all_improvement": -0.1, "t50_improvement": -0.2, "easy_degradation": 0.0},
                }
            }
        },
    }
    gate = u._gate(result)
    assert gate["passed"] == 7
    assert gate["total"] == 8
    assert gate["verdict"] == "stage42_u_ucy_endpoint_to_full_bridge_failed_blocker"
