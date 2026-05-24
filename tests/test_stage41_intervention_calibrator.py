from src import stage41_intervention_calibrator as cal


def test_stage41_intervention_calibrator_candidates() -> None:
    names = cal._candidate_base_trials()
    assert "all_agent_endpoint_risk_switch" in names
    assert len(names) >= 3


def test_stage41_intervention_score_penalizes_easy_harm() -> None:
    safe = {"all_improvement": 0.1, "t50_improvement": 0.1, "hard_failure_improvement": 0.1, "easy_degradation": 0.0, "harm_over_fallback": -0.1, "by_domain": {"a": {"all_improvement": 0.1}, "b": {"t50_improvement": 0.1}}}
    unsafe = dict(safe)
    unsafe["easy_degradation"] = 0.2
    assert cal._score_metrics(safe) > cal._score_metrics(unsafe)
