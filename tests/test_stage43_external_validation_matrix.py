from pathlib import Path

from src.stage43_external_validation_matrix import build_external_validation_matrix, run


def test_stage43_external_validation_matrix_gate_passes() -> None:
    payload = build_external_validation_matrix()
    gate = payload["stage43_at_gate"]
    assert gate["verdict"] == "stage43_at_external_validation_matrix_pass"
    assert gate["passed"] == gate["total"]


def test_stage43_external_validation_matrix_blocks_ungated_deployment() -> None:
    payload = build_external_validation_matrix()
    rows = {row["role"]: row for row in payload["comparison_rows"]}
    assert rows["ungated_neural_diagnostic"]["deployable"] is False
    assert rows["ungated_neural_diagnostic"]["metrics"]["easy_degradation"] > 0.02
    assert rows["source_safe_protected_neural"]["deployable"] is True
    assert payload["source_repair_summary"]["uniform_positive_per_source_claim_allowed"] is False


def test_stage43_external_validation_matrix_claim_boundary() -> None:
    payload = build_external_validation_matrix()
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    assert claim["true_3d_world_model"] is False
    assert claim["foundation_world_model"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False
    assert no_leakage["future_endpoint_input"] is False
    assert no_leakage["future_waypoint_input"] is False
    assert no_leakage["future_labels_eval_only"] is True


def test_stage43_external_validation_matrix_writes_reports() -> None:
    payload = run()
    assert payload["stage43_at_gate"]["passed"] == payload["stage43_at_gate"]["total"]
    assert Path("outputs/stage43_latent_state/stage43_external_validation_matrix.json").exists()
    assert Path("outputs/stage43_latent_state/stage43_external_validation_matrix.md").exists()
    assert Path("outputs/stage43_latent_state/stage43_stage_at_external_validation_matrix_gate.md").exists()
