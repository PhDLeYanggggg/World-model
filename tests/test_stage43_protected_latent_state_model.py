from __future__ import annotations

from src.stage43_protected_latent_state_model import main


_PAYLOAD = None


def _payload():
    global _PAYLOAD
    if _PAYLOAD is None:
        _PAYLOAD = main(["--quick", "--epochs", "1", "--batch-size", "1024", "--hidden-dim", "64", "--latent-dim", "16"])
    return _PAYLOAD


def test_stage43_protected_latent_training_runs_and_reports_gate():
    payload = _payload()
    gate = payload["stage43_c_gate"]
    assert gate["passed"] >= 6
    assert gate["gates"]["torch_training_fresh_run"]
    assert gate["gates"]["protected_eval_completed"]
    assert payload["result_source"] == "fresh_run"


def test_stage43_protected_latent_keeps_safety_claim_boundaries():
    payload = _payload()
    claim = payload["claim_boundary"]
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["ordinary_residual"] is False


def test_stage43_protected_latent_outputs_metrics():
    payload = _payload()
    metrics = payload["test_metrics_with_floor"]
    assert metrics["rows"] > 0
    assert "all_improvement_vs_floor" in metrics
    assert "t50_improvement_vs_floor" in metrics
    assert "easy_degradation_vs_floor" in metrics
    assert payload["latent_variance"] >= 0.0
