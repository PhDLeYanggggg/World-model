from __future__ import annotations

from src.stage14_pipeline import read_json
from src.stage17_pipeline import build_baseline_oracle, evaluate_baseline_selector, train_baseline_selector


def test_stage17_oracle_has_candidate_rows():
    metrics = build_baseline_oracle()
    assert metrics["rows"] > 0
    assert metrics["official_t50"]["count"] > 0
    assert metrics["official_t50"]["improvement"] >= 0.0


def test_stage17_selector_runs_and_reports_regret():
    train_baseline_selector()
    metrics = evaluate_baseline_selector()
    assert metrics["trained"] is True
    assert metrics["test_rows"] > 0
    assert metrics["selector_regret"] >= 0.0
    assert "choice_distribution" in metrics


def test_stage17_metrics_compare_to_global_baseline():
    metrics = read_json("outputs/reports/stage17_metrics.json", {})
    if not metrics:
        from src.stage17_pipeline import run_benchmark

        metrics = run_benchmark()
    assert "official_t50_selector_improvement" in metrics
    assert metrics["easy_degradation"] <= 1.0

