import numpy as np
import pytest

from src.evaluation.m3w_frozen_step_control import build


def test_one_and_two_recipient_matching_keep_their_original_choices():
    q = np.array([1, 1, 2, 2, 2, 3])
    cv = np.zeros((len(q), 12, 2))
    a, b = cv.copy(), cv.copy()
    a[[0, 2, 3], :, 0] = np.linspace(0, 2, 12)
    b[[1, 3, 4], :, 1] = np.linspace(2, 0, 12)
    result = build(cv, a, b, q)
    assert result['keep'].all()
    assert result['counts'] == dict(eligible_queries=2, collapsed_queries=0, one_edit_queries=1,
                                    two_edit_queries=1, original_edits=3, retained_edits=3)
    x, y = result['pairs']['matched_steps']
    norm = np.linalg.norm(x[[0, 2, 3]], axis=-1)
    np.testing.assert_allclose(norm, np.linalg.norm(y[[1, 3, 4]], axis=-1))
    assert np.all(x[:, :, 1] == 0) and np.all(y[:, :, 0] == 0)
    np.testing.assert_array_equal(result['pairs']['original'], np.stack((a, b)))


def test_collapsed_query_is_not_counted_as_agreement():
    cv = np.zeros((2, 12, 2))
    a, b = cv.copy(), cv.copy()
    a[0, 0, 0] = 1
    b[1, 1, 1] = 1
    out = build(cv, a, b, np.array([0, 0]))
    assert not out['keep'].any()
    assert out['counts']['collapsed_queries'] == 1 and out['counts']['retained_edits'] == 0


def test_missing_edits_rejected():
    cv = np.zeros((2, 12, 2))
    a = cv.copy()
    a[0, :, 0] = 1
    with pytest.raises(ValueError, match='count'):
        build(cv, a, cv, np.array([0, 0]))
