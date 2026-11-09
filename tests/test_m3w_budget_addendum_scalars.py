import pytest

from scripts.verify_m3w_budget_addendum_scalars import percentile, validate


def test_replay_keeps_an_unestimable_source_out_of_bootstrap():
    data = dict(counts={'arms': [[i, 2*i] for i in range(7)]},
                contrasts={'paired': [1., 2., None, 3., 4., 5., 6.]},
                resamples=[list(range(6))]*3000,
                expected={'arms': [21, 42], 'paired': [3.5, 3.5, 3.5]})
    assert validate(data) == data['expected']
    data['contrasts']['paired'][2] = 0.
    with pytest.raises(ValueError, match='must not be assigned zero'):
        validate(data)


def test_linear_interval_endpoints():
    assert percentile([0., 4., 8.], .25) == 2.
    assert percentile([0., 4., 8.], .975) == pytest.approx(7.8)
