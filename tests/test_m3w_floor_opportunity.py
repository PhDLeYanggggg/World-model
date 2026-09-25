import numpy as np
import pytest

from src.evaluation.m3w_floor_opportunity import floor_ledger, annotation_subsets
from src.evaluation.m3w_opportunity_diagnosis import causal_reasons


def ledger(cv, floor, neural, why, sites=None, roster=None):
    sites = np.array(['a']*len(cv)) if sites is None else np.asarray(sites)
    return floor_ledger(np.array(cv, float), np.array(floor, float), np.array(neural, float),
                        np.array(why, int), sites, expected_scenes=roster or ['a'], resamples=100)


def test_new_floor_regret_and_signed_fallback_accounting():
    r = ledger([10]*4, [8, 8, 12, 8], [6, 9, 20, 5], [5, 5, 4, 2])
    s = r['by_scene']['a']['sums']
    assert s['oracle_gain'] == 5
    assert s['captured_gain'] == 2 and s['selected_harm'] == 1
    assert s['missed_gain'] == 3 and s['missed_nonpositive_utility'] == 3
    assert s['fallback_regression'] == 2 and s['fallback_relief'] == 2
    assert s['original_gain'] == 1 and s['rebased_gain'] == 1
    assert s['signed_oracle_deficit'] == 4 and s['union_oracle_gain'] == 7
    assert r['summary']['original_gain']['equal_locality'] == pytest.approx(100/36)


def test_current_cv_can_beat_floor_candidate_oracle_so_deficit_is_signed():
    r = ledger([1], [10], [11], [4])
    s = r['by_scene']['a']['sums']
    assert s['original_gain'] == 9 and s['oracle_gain'] == 0
    assert s['signed_oracle_deficit'] == -9
    assert s['rebase_advantage'] == -9 and s['union_oracle_gain'] == 9


def test_unknown_labels_retain_population_and_decisions():
    r = ledger([2, np.nan], [1, np.nan], [0, np.nan], [5, 5])
    assert r['indexed_rows'] == 2 and r['supported_rows'] == 1 and r['unknown_rows'] == 1
    assert r['by_scene']['a']['counts']['switch'] == 2
    assert r['by_scene']['a']['switched_unknown_rows'] == 1


@pytest.mark.parametrize('case', ['missing_locality', 'zero_floor'])
def test_unavailable_localities_are_not_silently_removed(case):
    if case == 'missing_locality':
        r = ledger([1], [1], [0], [5], ['b'], ['a', 'b'])
    else:
        r = ledger([0, 1], [0, 1], [1, 0], [5, 5], ['a', 'b'], ['a', 'b'])
    assert r['summary']['oracle_gain']['equal_locality'] is None
    assert r['summary']['oracle_gain']['ci95'] is None


@pytest.mark.parametrize('bad', [[np.nan, 1], [-1, 1], [np.inf, 1]])
def test_invalid_or_misaligned_cost_support_is_rejected(bad):
    with pytest.raises(ValueError):
        ledger([1, 1], bad, [1, 1], [5, 5])


def test_future_poisoning_changes_ledger_not_causal_gate():
    utility = np.array([1., -1.]); risk = np.array([[1., .01], [1., .03]])
    why = causal_reasons(utility, risk, np.ones(2, bool), np.ones(2, bool))
    a = ledger([2, 2], [1, 1], [0, 3], why)
    b = ledger([2, 2], [1, 1], [3, 0], why)
    np.testing.assert_array_equal(why, causal_reasons(utility, risk, np.ones(2, bool), np.ones(2, bool)))
    assert a['summary']['original_gain'] != b['summary']['original_gain']


def test_annotation_subsets_preserve_unknowns_in_primary_and_are_eval_only():
    valid = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 0, 1]], bool)
    m = annotation_subsets(valid, np.array([1., 2., np.nan, 5.]), easy_cut=2, hard_cut=4)
    assert m['all'].all()
    np.testing.assert_array_equal(m['complete'], [True, False, False, False])
    np.testing.assert_array_equal(m['partial'], [False, True, False, True])
    np.testing.assert_array_equal(m['endpoint'], [True, False, False, True])
    assert m['easy_complete'].sum() == 1 and m['hard_complete'].sum() == 0


def test_invalid_subset_and_reason_are_rejected():
    with pytest.raises(ValueError):
        floor_ledger(np.ones(1), np.ones(1), np.zeros(1), np.array([5]), np.array(['a']),
                     expected_scenes=['a'], subset=np.array([1]))
    with pytest.raises(ValueError):
        ledger([1], [1], [0], [6])


def test_units_rescale_costs_not_relative_accounting():
    a = ledger([10, 2], [7, 3], [3, 5], [5, 4])
    b = ledger([100, 20], [70, 30], [30, 50], [5, 4])
    assert a['summary'] == b['summary']
