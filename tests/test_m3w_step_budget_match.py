import numpy as np
import pytest

from src.evaluation.m3w_step_budget_match import match


def test_matching_holds_edit_directions_and_same_reduced_peak():
    d = np.zeros((2, 2, 12, 2))
    d[0, :, :, 0] = np.linspace(1., 2., 12)
    d[1, :, :, 1] = np.linspace(2., 1., 12)
    x, r, s = match(d)
    assert .5 < s < 1.
    for k in ('smaller_peak', 'matched_steps'):
        assert np.allclose(np.linalg.norm(x[k], axis=-1).max(-1), r.max())
        assert np.array_equal(x[k] == 0., d == 0.)
        assert (np.linalg.norm(x[k], axis=-1) <= np.linalg.norm(d, axis=-1)).all()
    np.testing.assert_allclose(np.linalg.norm(x['matched_steps'], axis=-1), np.broadcast_to(r, (2, 2, 12)))
    assert not np.allclose(x['matched_steps'], x['smaller_peak'])


def test_no_common_positive_step_is_not_a_matched_edit():
    d = np.zeros((2, 2, 12, 2))
    d[0, :, :6, 0] = 1.
    d[1, :, 6:, 0] = 1.
    assert match(d) is None
    with pytest.raises(ValueError, match='not matched'):
        match(d*np.array([1., .5])[:, None, None, None])
