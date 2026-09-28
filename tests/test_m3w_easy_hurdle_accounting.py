import pytest
from scripts.analyze_m3w_easy_hurdle import exchange


def metric(benefit, harm, denominator=100):
    return dict(floor_error_sum=denominator, benefit_sum=benefit,
                positive_harm_sum=harm, error_sum=denominator-benefit+harm)


def test_harm_reduction_can_lose_more_benefit():
    result = exchange(metric(1, .5), metric(5, 2))
    assert result == dict(lost_benefit_pp=4, harm_reduction_pp=1.5,
                          net_gain_full_floor_pp=-2.5)


def test_identical_policies_have_zero_exchange():
    assert all(v == 0 for v in exchange(metric(3, 1), metric(3, 1)).values())


def test_accounting_refuses_mismatched_denominators_and_costs():
    with pytest.raises(ValueError):
        exchange(metric(1, 1, 0), metric(1, 1, 0))
    with pytest.raises(ValueError):
        exchange(metric(1, 1, 99), metric(1, 1))
    bad = metric(1, 1)
    bad['error_sum'] += 1
    with pytest.raises(AssertionError):
        exchange(bad, metric(1, 1))
