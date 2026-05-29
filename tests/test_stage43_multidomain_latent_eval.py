from __future__ import annotations

from pathlib import Path

import pytest

from src.stage43_multidomain_latent_eval import run_eval


def test_stage43_multidomain_eval_maps_current_split_blocker():
    if not Path("outputs/stage43_latent_state/checkpoints/stage43_protected_latent_small.pt").exists():
        pytest.skip("Stage43-C checkpoint is local and intentionally not committed.")
    payload = run_eval(bootstrap=20, batch_size=4096)
    gate = payload["stage43_e_gate"]
    assert gate["verdict"] == "stage43_e_multidomain_latent_eval_blocker_mapped"
    assert gate["multi_domain_latent_candidate"] is False
    assert payload["heldout_coverage"]["test_domains"] == ["UCY"]
    assert set(payload["heldout_coverage"]["missing_heldout_domains"]) >= {"ETH_UCY", "TrajNet"}
    assert payload["claim_boundary"]["multi_domain_latent_claim"] is False


def test_stage43_multidomain_eval_reports_seen_and_heldout_roles():
    if not Path("outputs/stage43_latent_state/checkpoints/stage43_protected_latent_small.pt").exists():
        pytest.skip("Stage43-C checkpoint is local and intentionally not committed.")
    payload = run_eval(bootstrap=20, batch_size=4096)
    assert payload["splits"]["train"]["role"] == "train_seen"
    assert payload["splits"]["val"]["role"] == "validation_seen"
    assert payload["splits"]["test"]["role"] == "heldout_test"
    assert payload["splits"]["test"]["metrics"]["rows"] > 0
