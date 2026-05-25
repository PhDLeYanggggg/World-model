from src import stage41_joint_distiller_multiseed as jdm


def test_summarize_metric_reports_min_max_mean():
    rows = [
        {"test_metrics": {"all_improvement": 0.1}},
        {"test_metrics": {"all_improvement": 0.3}},
        {"test_metrics": {"all_improvement": 0.5}},
    ]
    summary = jdm._summarize_metric(rows, "all_improvement")
    assert summary["min"] == 0.1
    assert summary["max"] == 0.5
    assert abs(summary["mean"] - 0.3) < 1e-12


def test_positive_domains_counts_any_positive_core_metric():
    metrics = {
        "by_domain": {
            "A": {"all_improvement": 0.0, "t50_improvement": 0.1, "hard_failure_improvement": 0.0},
            "B": {"all_improvement": -0.1, "t50_improvement": -0.1, "hard_failure_improvement": -0.1},
            "C": {"all_improvement": 0.0, "t50_improvement": 0.0, "hard_failure_improvement": 0.2},
        }
    }
    assert jdm._positive_domains(metrics) == 2
