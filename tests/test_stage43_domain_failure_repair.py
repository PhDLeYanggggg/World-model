from pathlib import Path

from src.stage43_domain_failure_repair import _trial_grid, build_domain_failure_repair


def test_stage43_domain_failure_repair_trial_grid_is_bounded() -> None:
    trials = _trial_grid(max_trials=30)
    assert 1 <= len(trials) <= 30
    assert any(trial["focus"] == "t50" for trial in trials)
    assert all(trial["test_tuned"] is False for trial in trials)


def test_stage43_domain_failure_repair_payload_boundaries() -> None:
    payload = build_domain_failure_repair(max_trials=4, bootstrap=5)
    assert payload["trial_count"] == 4
    assert payload["claim_boundary"]["test_threshold_tuning"] is False
    assert payload["claim_boundary"]["stage5c_executed"] is False
    assert payload["claim_boundary"]["smc_enabled"] is False
    assert payload["selected_policy"]["test_metrics"]["rows"] > 0


def test_stage43_domain_failure_repair_reports_weak_slices() -> None:
    payload = build_domain_failure_repair(max_trials=4, bootstrap=5)
    blocked = payload["selected_policy"]["blocked_slices_after_attempt"]
    assert blocked
    assert any("t50" in row["slice"] or "t100" in row["slice"] for row in blocked)


def test_stage43_domain_failure_repair_committed_report_exists_after_runner() -> None:
    assert Path("outputs/stage43_latent_state/stage43_external_validation_matrix.json").exists()
