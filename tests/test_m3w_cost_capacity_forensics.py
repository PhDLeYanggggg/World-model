import numpy as np
import pytest
from scripts.audit_m3w_cost_capacity import comparison_rows


def fixture():
    return dict(old_bits=np.array([True, False, True, False]),
        new_bits=np.array([False, True, True, True]),
        valid=np.array([[True, True], [True, True], [True, False], [False, False]]),
        cv=np.array([1., 2., 1., np.nan]), candidate=np.array([3., 1., 5., np.nan]),
        old_score=np.ones((4, 2)), new_score=np.zeros((4, 2)),
        distance=np.ones(4), speed=np.ones(4), easy_cut=1.5)


def test_fixed_groups_complete_costs_and_missing_outcomes():
    x = fixture()
    rows = comparison_rows(**x)
    lookup = {(r['subset'], r['group']): r for r in rows}
    added = lookup['all', 'new_only']
    assert added['selected'] == 2 and added['complete'] == 1 and added['unknown_ADE'] == 1
    assert added['costs']['realized_net_gain'] == 1
    assert lookup['all', 'overlap']['costs'] is None
    assert lookup['positive_easy', 'old_only']['costs']['realized_harm'] == 2
    assert lookup['positive_easy', 'new_all']['incomplete'] == 1


def test_outcomes_change_diagnostic_not_archived_choices():
    x = fixture()
    old, new = x['old_bits'].copy(), x['new_bits'].copy()
    first = comparison_rows(**x)
    x['candidate'][:3] += 100
    second = comparison_rows(**x)
    for a, b in zip(first, second):
        for field in ('selected', 'complete', 'unknown_ADE', 'incomplete'):
            assert a[field] == b[field]
    np.testing.assert_array_equal(old, x['old_bits'])
    np.testing.assert_array_equal(new, x['new_bits'])


def test_no_unknown_cut_or_non_boolean_choices():
    x = fixture()
    x['easy_cut'] = float('nan')
    with pytest.raises(ValueError):
        comparison_rows(**x)
    x['easy_cut'] = 1.5
    x['old_bits'] = x['old_bits'].astype(int)
    with pytest.raises(ValueError):
        comparison_rows(**x)
