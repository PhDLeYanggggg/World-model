import pytest

from src.evaluation.m3w_completion_contrasts import completion


def test_completion_recovers_upper_from_the_same_direct_pair_and_source_weights():
    x = completion([2., -1.], [-1., -3.], [[0, 0], [0, 1], [1, 1]])
    assert x['summary']['upper']['point'] == 3.
    assert x['summary']['lower']['point'] == -2.
    assert x['leave_one_out_envelopes'] == [[-3., 1.], [-1., 5.]]
    assert x['roster_sign'] == 'unresolved'
    assert x['leave_one_out_signs'] == ['unresolved', 'unresolved']


def test_sign_is_not_set_by_point_observed_and_rejects_invalid_debit():
    d = [[0, 1], [1, 0]]
    assert completion([-2., -4.], [-3., -6.], d)['roster_sign'] == 'negative'
    with pytest.raises(ValueError, match='aligned'):
        completion([-2., -4.], [-1., -6.], d)
