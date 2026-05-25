from src import stage41_teacher_guided_multiseed as multiseed


def test_candidate_eligible_requires_t100_and_collision() -> None:
    metrics = {
        "all_improvement": 0.1,
        "t50_improvement": 0.1,
        "t100_improvement": 0.0,
        "hard_failure_improvement": 0.1,
        "easy_degradation": 0.0,
    }
    assert not multiseed._candidate_eligible(metrics, collision_delta=0.0, switch_rate=0.2, ceiling=0.01)
    metrics["t100_improvement"] = 0.1
    assert multiseed._candidate_eligible(metrics, collision_delta=0.0, switch_rate=0.2, ceiling=0.01)
    assert not multiseed._candidate_eligible(metrics, collision_delta=0.2, switch_rate=0.2, ceiling=0.01)


def test_summarize_metric_uses_test_metrics() -> None:
    rows = [{"test_metrics": {"all_improvement": 0.1}}, {"test_metrics": {"all_improvement": 0.3}}]
    summary = multiseed._summarize_metric(rows, "all_improvement")
    assert summary["mean"] == 0.2
    assert summary["min"] == 0.1
    assert summary["max"] == 0.3
