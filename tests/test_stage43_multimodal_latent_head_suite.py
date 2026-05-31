from __future__ import annotations

from src import stage43_multimodal_latent_head_suite as y


def _payload() -> dict:
    return {
        "source": y.SOURCE,
        "preconditions": {
            "stage43_v_verdict": "stage43_v_world_state_head_audit_partial",
            "stage43_w_verdict": "stage43_w_density_proxy_repaired_validity_proxy_diagnostic",
            "stage43_x_verdict": "stage43_x_interaction_proxy_signal_validity_proxy_diagnostic",
        },
        "latent_state": {
            "min_variance": 0.1,
            "mean_variance": 0.5,
            "noncollapse_threshold": 0.01,
        },
        "head_suite": {
            "failure_risk": {"auroc": 0.86},
            "gain_opportunity": {"auroc": 0.87},
            "harm_guard": {"auroc": 0.90},
            "causal_history_density": {"r2": 0.81, "corr": 0.92},
            "future_interaction_risk": {"auroc": 0.76, "auprc": 0.32},
            "smoothness_validity_proxy": {"r2": 0.92},
        },
        "deployment_contract": {
            "physical_validity_true_claim_allowed": False,
            "diagnostic_only_heads": ["smoothness_validity_proxy"],
            "must_keep_safety_floor": True,
            "not_a_standalone_ungated_policy": True,
        },
        "no_leakage": {
            "future_waypoint_input": False,
            "future_labels_eval_or_supervision_only": True,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_accepts_complete_proxy_suite() -> None:
    gate = y._gate(_payload())
    assert gate["passed"] == gate["total"]
    assert gate["protected_multimodal_latent_state_candidate"] is True
    assert gate["standalone_world_model_deployable"] is False


def test_gate_rejects_collapsed_latent_state() -> None:
    payload = _payload()
    payload["latent_state"]["min_variance"] = 0.0
    gate = y._gate(payload)
    assert gate["passed"] < gate["total"]
    assert gate["protected_multimodal_latent_state_candidate"] is False


def test_gate_requires_safety_floor() -> None:
    payload = _payload()
    payload["deployment_contract"]["must_keep_safety_floor"] = False
    gate = y._gate(payload)
    assert gate["passed"] < gate["total"]
    assert gate["protected_multimodal_latent_state_candidate"] is False
