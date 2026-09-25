import numpy as np
import pytest
from src.evaluation.m3w_producer_transport import validate_roles, score_diagnosis


def test_transport_requires_complete_producer_exclusion():
    h, r = list('abcd'), list('efghijkl')
    validate_roles(h, h[:2], r)
    validate_roles(h, h, r)
    with pytest.raises(ValueError):
        validate_roles(h, ['a', 'e'], r)
    with pytest.raises(ValueError):
        validate_roles(h, h[:2], list('defghijk'))


def fixture(candidate):
    return score_diagnosis(np.array([2., 2., 4., np.nan]), np.array(candidate),
        np.array([[1., 0.], [1., 2.], [2., 0.], [1., 0.]]),
        np.array([[2., 0.], [2., 0.], [4., 1.], [1., 0.]]), np.ones(4, bool),
        np.array(['a', 'a', 'b', 'b']), easy_cut=2., event='all', resamples=10)


def test_future_labels_change_diagnosis_not_causal_decisions():
    a, b = fixture([1., 3., 2., np.nan]), fixture([3., 1., 6., np.nan])
    np.testing.assert_array_equal(a['reasons'], b['reasons'])
    assert a['ledger']['by_scene']['a']['contributions_percent']['net_gain'] == 25
    assert b['ledger']['by_scene']['a']['contributions_percent']['net_gain'] == -25
    assert a['ledger']['unknown_rows'] == 1
    assert a['score_errors']['b']['selected']['rows'] == 0


def test_paired_unknown_support_is_not_silently_changed():
    with pytest.raises(ValueError):
        fixture([1., 3., 2., 5.])


def test_event_target_and_native_unit_are_not_interchanged():
    a = fixture([1., 3., 2., np.nan])
    x = a['score_errors']['a']['population']
    assert x['utility']['actual_mean'] == [.5, .5]
    assert x['risk']['actual_mean'] == [2., .5]
    assert x['risk']['mae_over_mean_cv'] == [0., .25]
