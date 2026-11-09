from itertools import product

import pytest

from reproducibility.wepdtof_scalar_replay.verify import exact_interval, joint_state, source_means, supported


def test_risk_recomputed_from_three_required_components():
    assert supported([.02, .02, -.001])
    assert not supported([.01, None, -.001])
    assert not supported([.01, .01, .0200001])
    with pytest.raises(ValueError, match='negative'):
        supported([-.001, .02, 0.])


def test_dual_pass_requires_risk_and_two_positive_gains():
    row = dict(risk=[.01, .01, 0.], edits=1, unknown_edits=0, lower_vs_cv=2., lower_vs_ols4=0.)
    assert joint_state(row, 10.) == (True, False, 'risk_pass')
    assert joint_state({**row, 'lower_vs_ols4': .02}, 10.) == (True, True, 'risk_pass')
    assert joint_state({**row, 'risk': [None]*3, 'edits': 0}, 10.) == (False, False, 'no_intervention')


def test_source_mean_retains_group_only_if_all_views_are_estimable():
    assert source_means([[[1., 2.], [3., 4.]], [[4.], [None]], [[10.]]], lambda x: x) == [2.5, 10.]


def test_exact_bootstrap_matches_explicit_ordered_resamples():
    values = [-2., 0., 1., 5.]
    explicit = sorted(sum(a)/len(a) for a in product(values, repeat=len(values)))
    n = len(explicit)
    expected = [explicit[(n*q+39)//40-1] for q in (1, 39)]
    assert exact_interval(values) == expected
    assert exact_interval([0.]*10) == [0., 0.]
