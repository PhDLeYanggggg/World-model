import math

import pytest

from scripts.diagnose_m3w_ratio_ranking_estimand import counterexample


def test_realized_ratio_ranking_can_reverse_conditional_moment_risk():
    r = counterexample()
    assert r['conditional_risk_H_over_B']['A'] == pytest.approx(1/101)
    assert r['conditional_risk_H_over_B']['B'] == pytest.approx(.03)
    assert r['expected_realized_share']['A'] > r['expected_realized_share']['B']
    assert r['conditional_risk_H_over_B']['A'] < .02 < r['conditional_risk_H_over_B']['B']
    assert r['observed_share_pair_optimal_score_A_minus_B'] > 0
    assert r['cross_moment_pair_optimal_score_A_minus_B'] < 0


def test_existing_loss_can_prefer_underestimating_the_unsafe_state():
    r = counterexample()
    assert r['existing_rank_loss_biased_moments'] < r['existing_rank_loss_true_moments']
    assert r['original_supported_loss_equal']
    assert math.isfinite(r['existing_rank_loss_biased_moments'])
    assert r['proves_empirical_cause'] is False
    assert r['new_model_training'] is False
