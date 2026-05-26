from src import stage42_row_prediction_cache as r


def test_stage42_r_cache_report_and_claim_boundaries():
    result = r.run_stage42_row_prediction_cache()
    gate = result["stage42_r_gate"]
    assert gate["gates"]["row_prediction_cache_built"]
    assert gate["gates"]["combo_eval_from_cache"]
    assert gate["gates"]["validation_only_combo_selection"]
    assert gate["gates"]["no_leakage_pass"]
    assert gate["gates"]["stage5c_false"]
    assert gate["gates"]["smc_false"]
    assert result["claim_boundary"]["metric_or_seconds_claim"] is False


def test_stage42_r_cache_rows_are_not_committed_payloads():
    result = r.run_stage42_row_prediction_cache()
    assert result["cache_dir"] == str(r.CACHE_DIR)
    assert len(result["cache_rows"]) >= 3
    assert "arrays_for_bootstrap" not in result["rows"][0]


def test_stage42_r_core_metrics_are_reported():
    result = r.run_stage42_row_prediction_cache()
    summary = result["summary"]
    assert "ade_all" in summary
    assert "ade_t50" in summary
    assert "ade_hard_failure" in summary
    assert "ade_easy_degradation" in summary
