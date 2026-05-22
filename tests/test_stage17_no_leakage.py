from __future__ import annotations

from src.stage14_pipeline import read_json
from src.stage17_pipeline import train_baseline_selector


def test_stage17_selector_declares_no_leakage_inputs():
    model = train_baseline_selector()
    assert model["no_future_endpoint_input"] is True
    assert model["no_central_velocity"] is True
    assert model["no_test_endpoint_goals"] is True
    assert model["oracle_best_baseline_used_as_inference_input"] is False


def test_stage17_oracle_labels_are_diagnostic_not_inference_inputs():
    rows = read_json("data/stage17_baseline_oracle/oracle_rows.json", [])
    if not rows:
        from src.stage17_pipeline import build_baseline_oracle

        build_baseline_oracle()
        rows = read_json("data/stage17_baseline_oracle/oracle_rows.json", [])
    assert rows
    assert all(row["future_used_for_oracle_evaluation_only"] for row in rows[:25])
    assert all(row["oracle_best_baseline_used_as_inference_input"] is False for row in rows[:25])

