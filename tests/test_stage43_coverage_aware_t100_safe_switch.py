from __future__ import annotations

import numpy as np

from src import stage43_coverage_aware_t100_safe_switch as ci


class FakeSplit:
    def __init__(self) -> None:
        self.horizon = np.asarray([50, 100, 100])
        self.floor_ade = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        self.floor_fde = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
        self.hard = np.asarray([False, True, False])
        self.failure = np.asarray([False, False, False])
        self.easy = np.asarray([False, False, True])
        self.waypoint_valid = np.ones((3, 4), dtype=bool)
        self.waypoint_delta = np.zeros((3, 4, 2), dtype=np.float32)


def test_apply_t100_policy_overrides_base_t100_switch_to_floor() -> None:
    ds = FakeSplit()
    pred = {
        "waypoint": np.ones((3, 4, 2), dtype=np.float32),
        "gain": np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
        "harm": np.asarray([0.0, 0.8, 0.8], dtype=np.float32),
        "failure": np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
    }
    base_ade = np.asarray([0.5, 3.0, 3.0], dtype=np.float32)
    base_fde = np.asarray([1.0, 4.0, 4.0], dtype=np.float32)
    base_switch = np.asarray([True, True, True])
    selected_ade, selected_fde, switch = ci._apply_t100_policy(
        ds,
        pred,
        base_ade,
        base_fde,
        base_switch,
        {"gain_threshold": 1.01, "harm_threshold": -0.01, "failure_threshold": 1.01},
    )
    assert selected_ade.tolist() == [0.5, 1.0, 1.0]
    assert selected_fde.tolist() == [1.0, 2.0, 2.0]
    assert switch.tolist() == [True, False, False]


def _payload(t100: float, easy: float = 0.0) -> dict:
    metrics = {
        "rows": 100,
        "full_waypoint_ade_improvement_vs_floor": 0.3,
        "endpoint_fde_improvement_vs_floor": 0.3,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.2,
        "t50_endpoint_fde_improvement_vs_floor": 0.2,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": t100,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.2,
        "easy_degradation_vs_floor": easy,
        "switch_rate": 0.4,
    }
    return {
        "coverage_aware_latent_dynamics": {
            "verdict": "stage43_cg_coverage_aware_latent_dynamics_candidate_pass",
            "mode": "medium",
        },
        "t100_failure_audit": {"verdict": "stage43_ch_t100_failure_audit_pass_blocker_confirmed"},
        "result_source": ci.SOURCE,
        "training_protocol": {"selection_data": "validation_only", "test_threshold_tuning": False},
        "no_leakage": {
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "base_cg_test_metrics": {**metrics, "t100_raw_frame_full_waypoint_diagnostic_vs_floor": -0.05},
        "test_metrics_with_t100_safe_switch": metrics,
        "test_by_horizon": {
            "100": {
                "rows": 10,
                "full_waypoint_ade_improvement_vs_floor": t100,
                "endpoint_fde_improvement_vs_floor": t100,
                "easy_degradation_vs_floor": easy,
                "switch_rate": 0.0,
            }
        },
        "deployment_decision": {"t100_latent_switch_deployable": t100 > 0.0},
    }


def test_gate_passes_floor_repair_when_t100_is_nonnegative_not_positive() -> None:
    gate = ci._gate(_payload(0.0))
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_ci_t100_safe_switch_pass_floor_repair"
    assert gate["deploy_t100_latent_switch"] is False
    assert gate["deploy_t100_safe_floor_repair"] is True


def test_gate_fails_if_t100_remains_negative() -> None:
    gate = ci._gate(_payload(-0.01))
    assert gate["gates"]["t100_negative_repaired_to_nonnegative"] is False
    assert gate["verdict"] == "stage43_ci_t100_safe_switch_incomplete"
