from __future__ import annotations

from pathlib import Path

import pytest

from src.stage43_unit_consistent_safe_switch import PRIOR_EASY_GUARD, run_safe_switch


def test_stage43_i_fixed_prior_safe_switch_repairs_easy_harm():
    if not Path("outputs/stage43_latent_state/checkpoints/stage43_source_level_latent_full.pt").exists():
        pytest.skip("Stage43-G checkpoint is local and intentionally not committed.")
    payload = run_safe_switch(bootstrap=20, batch_size=4096)
    dep = payload["deployment_policy"]
    assert dep["test_tuned"] is False
    assert dep["name"] == "domain_capped_prior_easy_guard"
    assert dep["policy"]["stage35_easy_prob_max"] == PRIOR_EASY_GUARD
    assert dep["test_metrics"]["easy_degradation_vs_floor"] <= 0.02
    assert max(row["easy_degradation_vs_floor"] for row in dep["domain_metrics"].values()) <= 0.02
    assert dep["test_metrics"]["switch_rate"] > 0.0
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False


def test_stage43_i_reports_diagnostic_stage43g_failure():
    if not Path("outputs/stage43_latent_state/checkpoints/stage43_source_level_latent_full.pt").exists():
        pytest.skip("Stage43-G checkpoint is local and intentionally not committed.")
    payload = run_safe_switch(bootstrap=20, batch_size=4096)
    table = {row["name"]: row for row in payload["policy_table"]}
    assert "stage43g_validation_policy_diagnostic" in table
    assert table["stage43g_validation_policy_diagnostic"]["test_metrics"]["easy_degradation_vs_floor"] > 0.02
    assert payload["stage43_i_gate"]["gates"]["no_metric_seconds_stage5c_smc_claim"] is True
