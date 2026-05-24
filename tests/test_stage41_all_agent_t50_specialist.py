import numpy as np

from src import stage41_all_agent_t50_specialist as t50


def test_stage41_t50_domain_onehot_shape() -> None:
    out = t50._domain_onehot(np.array(["ETH_UCY", "UCY", "missing"]))
    assert out.shape == (3, len(t50.DOMAINS))
    assert out[0].sum() == 1.0
    assert out[2].sum() == 0.0


def test_stage41_t50_metric_score_rewards_t50() -> None:
    better = {"all_improvement": 0.0, "t50_improvement": 0.1, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "harm_over_fallback": 0.0}
    worse = {"all_improvement": 0.0, "t50_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "harm_over_fallback": 0.0}
    assert t50._metric_score(better) > t50._metric_score(worse)
