from src import stage41_composite_tail_multiseed as multiseed


def test_metric_delta_reports_improvement_vs_teacher():
    current = {
        "all_improvement": 0.12,
        "t50_improvement": 0.08,
        "t100_improvement": 0.09,
        "hard_failure_improvement": 0.11,
        "easy_degradation": 0.0,
    }
    teacher = {
        "all_improvement": 0.10,
        "t50_improvement": 0.07,
        "t100_improvement": 0.08,
        "hard_failure_improvement": 0.09,
        "easy_degradation": 0.0,
    }
    delta = multiseed._metric_delta(current, teacher)
    assert delta["all_delta"] > 0
    assert delta["t50_delta"] > 0
    assert delta["hard_delta"] > 0
    assert delta["easy_delta"] == 0.0


def test_summarize_nested_metric_min_max():
    rows = [
        {"test_metrics": {"all_improvement": 0.1}},
        {"test_metrics": {"all_improvement": 0.2}},
        {"test_metrics": {"all_improvement": 0.3}},
    ]
    summary = multiseed._summarize(rows, ["test_metrics", "all_improvement"])
    assert summary["min"] == 0.1
    assert summary["max"] == 0.3
    assert abs(summary["mean"] - 0.2) < 1e-12
