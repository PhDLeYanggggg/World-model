from __future__ import annotations

import numpy as np

from src import stage43_coverage_aware_t100_causal_feature_repair as ck


class FakeSplit:
    def __init__(self) -> None:
        self.x = np.zeros((2, 3), dtype=np.float32)
        self.floor_waypoint_delta = np.ones((2, 4, 2), dtype=np.float32)


def test_causal_features_exclude_label_error_diagnostics() -> None:
    ds = FakeSplit()
    pred = {
        "waypoint": np.zeros((2, 4, 2), dtype=np.float32),
        "latent": np.zeros((2, 5), dtype=np.float32),
        "gain": np.zeros(2, dtype=np.float32),
        "harm": np.zeros(2, dtype=np.float32),
        "failure": np.zeros(2, dtype=np.float32),
        "density": np.zeros(2, dtype=np.float32),
    }
    features = ck._make_causal_features(ds, pred)  # type: ignore[arg-type]
    # x + floor waypoint delta + latent candidate waypoint + latent + four scalar heads.
    assert features.shape == (2, 3 + 8 + 8 + 5 + 4)


def _payload(prior_deployed: bool, t100: float) -> dict:
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
        "prior_stage43_cj_audit": {
            "prior_stage43_cj_label_derived_features_in_specialist_input": True,
            "deployment_contamination": prior_deployed,
        },
        "stage43_ci_precondition": {"verdict": "stage43_ci_t100_safe_switch_pass_floor_repair"},
        "result_source": ck.SOURCE,
        "checkpoint": "outputs/stage43_latent_state/checkpoints/fake.pt",
        "checkpoint_committed": False,
        "causal_feature_contract": {"excluded_label_derived_eval_error_features": True},
        "training_protocol": {"selection_data": "validation_only", "test_threshold_tuning": False},
        "no_leakage": {
            "label_derived_eval_error_features_in_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "ci_floor_test_metrics": {**metrics, "t100_raw_frame_full_waypoint_diagnostic_vs_floor": 0.0},
        "test_metrics_with_causal_specialist": metrics,
        "deployment_decision": {"deploy_t100_causal_specialist": t100 > 0.0},
    }


def test_gate_passes_keep_floor_when_prior_leaky_diagnostic_was_not_deployed(monkeypatch) -> None:
    monkeypatch.setattr(ck.Path, "exists", lambda self: True)
    gate = ck._gate(_payload(False, 0.0))
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_ck_t100_causal_feature_repair_pass_keep_ci_floor"
    assert gate["deploy_t100_causal_specialist"] is False


def test_gate_fails_if_prior_leaky_specialist_was_deployed(monkeypatch) -> None:
    monkeypatch.setattr(ck.Path, "exists", lambda self: True)
    gate = ck._gate(_payload(True, 0.05))
    assert gate["passed"] == gate["total"] - 1
    assert gate["gates"]["prior_cj_not_deployed_or_flagged"] is False
    assert gate["deploy_t100_causal_specialist"] is False
