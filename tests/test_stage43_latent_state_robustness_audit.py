from __future__ import annotations

from pathlib import Path

import pytest

from src.stage43_latent_state_robustness_audit import run_audit


def test_stage43_latent_robustness_audit_smoke():
    if not Path("outputs/stage43_latent_state/checkpoints/stage43_protected_latent_small.pt").exists():
        pytest.skip("Stage43-C local checkpoint is intentionally not committed.")
    payload = run_audit(bootstrap_n=50, max_test=2000, batch_size=1024)
    gate = payload["stage43_d_gate"]
    assert gate["gates"]["stage43_c_checkpoint_exists"]
    assert payload["metrics"]["rows"] == 2000
    assert payload["bootstrap"]["all"]["bootstrap_n"] == 50
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
    assert payload["claim_boundary"]["metric_or_seconds_claim"] is False


def test_stage43_latent_robustness_records_scope_limitation():
    if not Path("outputs/stage43_latent_state/checkpoints/stage43_protected_latent_small.pt").exists():
        pytest.skip("Stage43-C local checkpoint is intentionally not committed.")
    payload = run_audit(bootstrap_n=50, max_test=2000, batch_size=1024)
    assert payload["domain_scope"]["test_domains"] == ["UCY"]
    assert payload["domain_scope"]["multi_domain_test"] is False
    assert payload["stage43_d_gate"]["multi_domain_claim_allowed"] is False
