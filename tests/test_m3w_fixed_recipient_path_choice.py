import numpy as np
import pytest

from src.evaluation.m3w_fixed_recipient_path_choice import select


def test_path_choice_cannot_change_recipient_set_or_candidate_order():
    row = np.array([0, 0, 1, 1, 2])
    i, j = np.array([0, 1, 0, 2]), np.array([2, 2, 3, 4])
    s = select(row, i, j, {'independent': np.array([1., 1., .5, 100.]),
                           'joint': np.array([1., 2., 3., 100.])}, [1, 0])
    np.testing.assert_array_equal(s['independent'][0], [0, 2])
    np.testing.assert_array_equal(s['joint'][0], [0, 3])
    with pytest.raises(ValueError, match='no legal'):
        select(row, i, j, {'joint': np.ones(4)}, [0, 2])
