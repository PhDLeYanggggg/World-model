from __future__ import annotations

import numpy as np

from src import stage43_coverage_aware_t100_long_horizon_specialist as cj


class FakeBase:
    def __init__(self) -> None:
        self.horizon = np.asarray([50, 100, 100])
        self.floor_ade = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        self.floor_fde = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
        self.floor_waypoint_delta = np.zeros((3, 4, 2), dtype=np.float32)
        self.waypoint_delta = np.zeros((3, 4, 2), dtype=np.float32)
        self.waypoint_valid = np.ones((3, 4), dtype=bool)
        self.hard = np.asarray([False, True, False])
        self.failure = np.asarray([False, False, False])
        self.easy = np.asarray([False, False, True])
        self.y_gain = np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
        self.y_harm = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def _split() -> cj.SpecialistSplit:
    base = FakeBase()
    return cj.SpecialistSplit(
        base=base,  # type: ignore[arg-type]
        features=np.zeros((3, 5), dtype=np.float32),
        target_residual=np.zeros((3, 4, 2), dtype=np.float32),
        valid=np.ones((3, 4), dtype=np.float32),
        cg_candidate_ade=np.ones(3, dtype=np.float32),
        cg_candidate_fde=np.ones(3, dtype=np.float32),
        ci_ade=np.asarray([0.5, 1.0, 1.0], dtype=np.float32),
        ci_fde=np.asarray([1.0, 2.0, 2.0], dtype=np.float32),
        ci_switch=np.asarray([True, False, False]),
    )


def test_apply_policy_only_switches_t100_when_gain_harm_allow() -> None:
    split = _split()
    pred = {
        "residual": np.asarray(
            [
                [[[1.0, 0.0]]] * 4,
                [[[0.0, 0.0]]] * 4,
                [[[1.0, 0.0]]] * 4,
            ],
            dtype=np.float32,
        ).reshape(3, 4, 2),
        "gain": np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
        "harm": np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
    }
    selected_ade, selected_fde, switched, _, _ = cj._apply_policy(
        split,
        pred,
        {"gain_threshold": 0.5, "harm_threshold": 0.1},
    )
    assert selected_ade.tolist() == [0.5, 0.0, 1.0]
    assert selected_fde.tolist() == [1.0, 0.0, 2.0]
    assert switched.tolist() == [True, True, False]


def _payload(t100: float) -> dict:
    metrics = {
        "rows": 100,
        "full_waypoint_ade_improvement_vs_floor": 0.2,
        "endpoint_fde_improvement_vs_floor": 0.2,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.2,
        "t50_endpoint_fde_improvement_vs_floor": 0.2,
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": t100,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.2,
        "easy_degradation_vs_floor": 0.0,
        "switch_rate": 0.4,
    }
    return {
        "stage43_ci_precondition": {"verdict": "stage43_ci_t100_safe_switch_pass_floor_repair"},
        "result_source": cj.SOURCE,
        "checkpoint": "outputs/stage43_latent_state/checkpoints/fake.pt",
        "checkpoint_committed": False,
        "training_protocol": {"selection_data": "validation_only", "test_threshold_tuning": False},
        "no_leakage": {
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "ci_floor_test_metrics": {**metrics, "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0},
        "test_metrics_with_specialist": metrics,
        "deployment_decision": {"deploy_t100_specialist": t100 > 0.0},
    }


def test_gate_can_pass_honest_keep_floor_when_t100_not_positive(monkeypatch) -> None:
    monkeypatch.setattr(cj.Path, "exists", lambda self: True)
    gate = cj._gate(_payload(0.0))
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cj_t100_long_horizon_specialist_pass_keep_ci_floor"
    assert gate["deploy_t100_specialist"] is False


def test_gate_marks_positive_t100_success(monkeypatch) -> None:
    monkeypatch.setattr(cj.Path, "exists", lambda self: True)
    gate = cj._gate(_payload(0.05))
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_cj_t100_long_horizon_specialist_pass_positive_t100"
    assert gate["deploy_t100_specialist"] is True
