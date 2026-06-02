from __future__ import annotations

import numpy as np

from src import stage43_latent_adapter_downstream_heads as m


def _payload(*, adapter_ade: float = 0.8, identity_ade: float = 1.0, m_ade: float = 0.9) -> dict:
    protected = {
        "rows": 10,
        "full_waypoint_ade_improvement_vs_floor": 0.01,
        "t50_full_waypoint_ade_improvement_vs_floor": 0.02,
        "hard_failure_full_waypoint_ade_improvement_vs_floor": 0.03,
        "easy_degradation_vs_floor": 0.0,
    }
    return {
        "stage43_bz_precondition": {"verdict": "stage43_bz_latent_transition_adapter_repair_pass"},
        "protocol": {"train_only_heads": True},
        "no_leakage": {
            "future_labels_as_inputs": False,
            "future_labels_train_eval_only": True,
            "test_threshold_tuning": False,
        },
        "variants": {
            "identity_z_t": {
                "eval": {"mean_ade": identity_ade, "risk": {"mean_defined_auroc": 0.7}},
                "protected": {"test_metrics_with_floor": dict(protected)},
            },
            "stage43_m_z_next": {
                "eval": {"mean_ade": m_ade, "risk": {"mean_defined_auroc": 0.75}},
                "protected": {"test_metrics_with_floor": dict(protected)},
            },
            "stage43_bz_adapter_z_next": {
                "eval": {"mean_ade": adapter_ade, "risk": {"mean_defined_auroc": 0.8}},
                "protected": {
                    "test_metrics_with_floor": dict(protected),
                    "slice_summary": {"domain": {"UCY": {}}, "horizon": {"50": {}}},
                },
            },
        },
        "best_overall_variant_by_validation_objective": "stage43_bz_adapter_z_next",
        "best_adapter_variant_by_validation_objective": "stage43_bz_adapter_z_next",
        "selected_adapter_variant": "stage43_bz_adapter_z_next",
        "claim_boundary": {"metric_or_seconds_claim": False, "stage5c_executed": False, "smc_enabled": False},
        "long_objective_complete": False,
    }


def test_ridge_fit_and_apply_shapes() -> None:
    x = np.eye(4, dtype=np.float32)
    y = np.ones((4, 2), dtype=np.float32)
    weights = m._fit_ridge(x, y, ridge=1e-2)
    pred = m._apply_linear(x, weights)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_predict_heads_clips_risk_scores_and_shapes_waypoints() -> None:
    latent = np.ones((3, 2), dtype=np.float32)
    weights = {
        "waypoint": np.zeros((3, 8), dtype=np.float32),
        "risk": np.ones((3, 3), dtype=np.float32) * 10.0,
        "density": np.ones((3, 1), dtype=np.float32) * -10.0,
    }
    pred = m._predict_heads(latent, weights)
    assert pred["waypoint"].shape == (3, 4, 2)
    assert np.all((pred["failure"] >= 0.0) & (pred["failure"] <= 1.0))
    assert np.all((pred["density"] >= 0.0) & (pred["density"] <= 1.0))


def test_gate_passes_when_adapter_heads_beat_identity_and_stage43m() -> None:
    gate = m._gate(_payload(adapter_ade=0.8, identity_ade=1.0, m_ade=0.9))
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_ca_latent_adapter_downstream_heads_pass"


def test_gate_reports_partial_lift_if_adapter_only_beats_identity() -> None:
    payload = _payload(adapter_ade=0.95, identity_ade=1.0, m_ade=0.9)
    payload["variants"]["stage43_bz_adapter_z_next"]["eval"]["risk"]["mean_defined_auroc"] = 0.72
    gate = m._gate(payload)
    assert gate["gates"]["adapter_waypoint_ungated_beats_identity"] is True
    assert gate["gates"]["adapter_waypoint_ungated_beats_stage43_m"] is False
    assert gate["verdict"] == "stage43_ca_latent_adapter_downstream_heads_partial_lift"
