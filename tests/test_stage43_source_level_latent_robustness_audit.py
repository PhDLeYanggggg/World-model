from __future__ import annotations

from pathlib import Path

import pytest

from src.stage43_source_level_latent_robustness_audit import run_audit


def test_stage43_h_unit_consistency_audit_detects_stage43g_overclaim():
    if not Path("outputs/stage43_latent_state/checkpoints/stage43_source_level_latent_full.pt").exists():
        pytest.skip("Stage43-G checkpoint is local and intentionally not committed.")
    payload = run_audit(bootstrap=20, batch_size=4096)
    assert payload["unit_consistency"]["stage43_g_normalized_metrics_not_deployment_evidence"] is True
    assert payload["unit_consistency"]["normalized_all_minus_unit_all"] > 0.1
    assert payload["stage43_h_gate"]["keep_frozen_floor"] is True


def test_stage43_h_reports_unit_metrics_and_proximity_proxy():
    if not Path("outputs/stage43_latent_state/checkpoints/stage43_source_level_latent_full.pt").exists():
        pytest.skip("Stage43-G checkpoint is local and intentionally not committed.")
    payload = run_audit(bootstrap=20, batch_size=4096)
    assert payload["unit_consistent_metrics"]["rows"] >= 80000
    assert "near_005_delta_vs_floor" in payload["proximity"]
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
