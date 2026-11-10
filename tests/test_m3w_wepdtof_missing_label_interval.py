import pytest

from scripts.audit_m3w_wepdtof_missing_label_interval import analyze, group_bounds, verdict


def make_group(*values):
    return [[{'paired': {'fixed_independent': v}} for v in values]]


def test_reflected_debit_and_original_missing_group_rule():
    data = {'groups': [make_group((3., 2.), (1., -1.)), make_group((1., .5), (None, None))]}
    assert group_bounds(data, 'fixed_independent') == [(.5, 2., 3.5)]
    with pytest.raises(ValueError, match='Invalid archived contrast'):
        group_bounds({'groups': [make_group((1., 2.))]}, 'fixed_independent')


@pytest.mark.parametrize(('a', 'b', 'expected'), [
    (1., 2., 'positive_under_all_missing_completions'),
    (-2., -1., 'negative_under_all_missing_completions'),
    (0., 0., 'score_tie'),
    (-1., 2., 'sign_not_resolved_by_bounds'),
])
def test_sign_is_not_inferred_from_one_endpoint(a, b, expected):
    assert verdict(a, b) == expected


def test_identification_and_bootstrap_intervals_are_not_interchangeable():
    groups = [(-4., -2., 0.), (0., 2., 4.)]
    result = analyze(groups, 3, {'2': [0, 0, 0, 1, 1, 0, 1, 1]*750})
    assert result['identification_envelope_pp'] == [-2., 2.]
    assert result['empirical_bootstrap_outer_pp'] == [-4., 4.]
    assert result['estimates']['observed']['estimable_sources'] == 2
