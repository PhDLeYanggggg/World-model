import numpy as np
import pytest

from src.evaluation.m3w_step_visibility_rank import normalized_debit


def test_weighting_moves_the_debit_to_missing_steps_not_the_whole_track():
    distance = np.ones((1, 2, 12))
    distance[0, 0, 7] = 3.
    a = np.zeros_like(distance, dtype=bool)
    a[0, 0, 7] = True
    b = np.zeros_like(a)
    b[0, 0, 0] = True
    np.testing.assert_allclose(normalized_debit(distance, a), [3/48])
    np.testing.assert_allclose(normalized_debit(distance, b), [1/48])
    np.testing.assert_allclose(normalized_debit(distance*2, a), [3/48])


def test_debit_rejects_invalid_probability_and_no_edit():
    distance = np.ones((1, 2, 12))
    with pytest.raises(ValueError, match='valid probabilities'):
        normalized_debit(distance, 1.01*distance)
    with pytest.raises(ValueError, match='differ from the fallback'):
        normalized_debit(distance*0., distance)
