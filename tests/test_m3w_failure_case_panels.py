import numpy as np
import pytest

from src.evaluation.m3w_failure_case_panels import compare, median_case


def test_cumulative_debit_does_not_treat_missing_steps_as_zero_error():
    left = np.zeros((2, 12, 2))
    right = left.copy()
    right[0, :, 0] = 1.
    mask = np.ones((2, 12), dtype=bool)
    mask[0, 5:] = False
    y = np.where(mask[..., None], right, np.nan)
    out = compare(left, right, y, mask, 1.)
    assert out['observed'][-1] == 5.
    assert out['lower'][-1] == -2.
    assert out['upper'][-1] == 12.
    assert sum(out['actor_observed']) == 5.
    assert sum(out['actor_debit']) == 7.


def test_case_is_a_stably_selected_median_not_an_extreme():
    x = [dict(rank=i, member=j) for i, j in [(4, 'd'), (2, 'b'), (9, 'a'), (2, 'a')]]
    assert median_case(x) == dict(rank=2, member='b')
    assert median_case([]) is None


def test_swapping_proposals_reverses_gain_and_exchanges_bounds():
    l = np.zeros((2, 12, 2))
    r = np.ones_like(l)
    v = np.ones((2, 12), dtype=bool)
    v[0, 4:8] = False
    a, b = (compare(x, z, r, v, .4) for x, z in ((l, r), (r, l)))
    np.testing.assert_allclose(a['observed'], -np.array(b['observed']))
    np.testing.assert_allclose(a['upper'], -np.array(b['lower']))
    with pytest.raises(ValueError, match='Invalid matched'):
        compare(l, r, r, v.astype(float), .4)
