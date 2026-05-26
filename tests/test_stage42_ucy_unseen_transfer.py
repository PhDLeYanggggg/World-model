from src import stage42_ucy_unseen_transfer as t


def test_stage42_t_reports_ucy_blocker_without_overclaim():
    result = t.run_stage42_ucy_unseen_transfer()
    assert result["stage42_t_gate"]["verdict"] in {
        "stage42_t_ucy_transfer_blocked_no_candidate_predictions",
        "stage42_t_ucy_unseen_transfer_partial",
        "stage42_t_ucy_unseen_transfer_pass",
    }
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False
    assert result["claim_boundary"]["stage5c_executed"] is False
    assert result["claim_boundary"]["smc_enabled"] is False


def test_stage42_t_uses_validation_only_transfer_rule():
    result = t.run_stage42_ucy_unseen_transfer()
    assert result["transfer_rule"]["source"] == "fresh_run_validation_only_unseen_domain_transfer_rule"
    assert result["no_leakage"]["ucy_test_used_for_policy_selection"] is False
    assert result["stage42_t_gate"]["gates"]["ucy_test_evaluated_once"]


def test_stage42_t_available_source_oracle_is_reported():
    result = t.run_stage42_ucy_unseen_transfer()
    oracle = result["available_source_oracle"]
    assert oracle["source"] == "fresh_run_test_diagnostic_only_available_source_oracle"
    assert "any_available_nonfloor_prediction" in oracle
