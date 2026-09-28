import inspect
import numpy as np
import pytest

from src.world_model.m3w_fitting_switch_diagnostic import causal_screens, diagnose


def example():
    n = 6
    return dict(ids=np.arange(n), sites=np.array(['A']*3+['B']*3),
        recordings=np.array(['r']*n), frames=np.zeros(n, int),
        utility=np.array([3., 2., 1., 1., 2., 3.]), moving=np.ones(n, bool),
        supported=np.ones(n, bool), risks={'raw': np.array([[-1., -1.], [1., -1.],
            [1., -1.], [-1., -1.], [1., -1.], [1., -1.]])},
        known=np.ones(n, bool), easy=np.ones(n), reference=np.ones(n),
        benefit=np.array([0., 2., 1., 0., 2., 1.]), harm=np.array([1., 0., 0., 1., 0., 0.]))


def test_label_free_screen_and_known_ranking_counterexample():
    assert set(inspect.signature(causal_screens).parameters) == {'utility', 'moving', 'supported', 'risks'}
    d = diagnose(**example())
    r = d['arms']['raw']
    assert r['queries']['all']['violating'] == 2
    assert r['queries']['easy']['violating'] == 2
    assert r['ranking']['queries_informative'] == 2
    assert r['ranking']['oracle_minus_utility_sum'] == pytest.approx(4.)
    assert r['ranking']['oracle_minus_screen_sum'] == pytest.approx(6.)
    assert r['row_screen']['net_gain'] == pytest.approx(-1/3)
    assert sum(v['benefit'] for v in r['partition'].values()) == pytest.approx(d['total']['benefit'])


def test_unknown_rows_are_not_used_to_select_or_impute_truth():
    x = example(); before = causal_screens(x['utility'], x['moving'], x['supported'], x['risks'])
    x['known'][0] = False
    for key in ('easy', 'reference', 'benefit', 'harm'): x[key][0] = np.nan
    d = diagnose(**x)
    np.testing.assert_array_equal(before['raw']['admitted'], [True, False, False, True, False, False])
    r = d['arms']['raw']
    assert r['queries']['all']['unknown_selected'] == 1
    assert r['queries']['all']['violating'] == 1
    assert r['ranking']['queries_incomplete'] == 1
    assert d['unknown_rows'] == 1


def test_empty_selection_is_undefined_not_safe():
    x = example(); x['utility'][:] = -1
    d = diagnose(**x)['arms']['raw']
    assert d['queries']['all']['undefined'] == 2
    assert d['queries']['all']['defined'] == 0
    assert d['queries']['easy']['undefined'] == 2
    assert d['ranking']['queries_zero_count'] == 2


def test_reason_partition_is_exclusive_and_causal():
    x = example(); x['moving'][0] = False; x['supported'][1] = False; x['utility'][2] = 0
    x['risks']['raw'][4] = [-1, 1]
    a = causal_screens(x['utility'], x['moving'], x['supported'], x['risks'])['raw']
    assert np.all(sum(a[k].astype(int) for k in a) == 1)
    assert [int(a[k].sum()) for k in a] == [1, 1, 1, 1, 1, 1]


def test_grouped_means_do_not_overweight_crowded_sources():
    x = example(); d = diagnose(**x)
    assert d['total']['benefit'] == pytest.approx(1)
    assert set(d['by_site']) == {'A', 'B'}
    for site in ('A', 'B'): assert d['by_site'][site]['total']['benefit'] == pytest.approx(1)


def test_invalid_alignment_and_partial_labels_rejected():
    x = example(); x['ids'][1] = x['ids'][0]
    with pytest.raises(ValueError): diagnose(**x)
    x = example(); x['benefit'][0] = np.nan
    with pytest.raises(ValueError): diagnose(**x)
    x = example(); x['harm'][1] = 1
    with pytest.raises(ValueError): diagnose(**x)


def test_no_easy_denominator_is_not_zero_risk():
    x = example(); x['easy'][:] = 0
    d = diagnose(**x)['arms']['raw']
    assert d['queries']['easy']['undefined'] == 2
    assert d['row_screen']['easy_selected_positive_harm_ratio'] is None


def test_score_ties_are_broken_by_stable_row_id():
    x = example(); x['utility'][:] = 1
    a = diagnose(**x)['arms']['raw']['ranking']
    order = np.array([2, 0, 1, 5, 3, 4])
    y = {k: ({a: b[order] for a, b in v.items()} if k == 'risks' else v[order]) for k, v in x.items()}
    b = diagnose(**y)['arms']['raw']['ranking']
    assert a == b
