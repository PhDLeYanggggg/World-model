from __future__ import annotations

from src.stage43_source_level_heldout_split import build_source_level_split


def test_stage43_source_level_split_contains_all_domains_in_test():
    payload = build_source_level_split()
    assert payload["stage43_f_gate"]["verdict"] == "stage43_f_source_level_split_ready"
    assert set(payload["pool"]["domains"]) >= {"ETH_UCY", "TrajNet", "UCY"}
    assert set(payload["split_summary"]["test"]["domains"]) >= {"ETH_UCY", "TrajNet", "UCY"}
    assert payload["no_leakage"]["source_file_disjoint"] is True
    assert payload["no_leakage"]["row_overlap_pass"] is True


def test_stage43_source_level_split_records_old_checkpoint_boundary():
    payload = build_source_level_split()
    boundary = payload["claim_boundary"]
    assert boundary["old_split_pool_reused_for_new_stage43_split"] is True
    assert boundary["previous_stage43_checkpoint_not_official_for_new_split"] is True
    assert boundary["new_training_or_evaluation_not_run"] is True
    assert boundary["stage5c_executed"] is False
    assert boundary["smc_enabled"] is False
