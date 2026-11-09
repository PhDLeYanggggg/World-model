import numpy as np
import pytest

from src.world_model.m3w_counterfactual_support import loss_events


def test_only_removing_the_last_reference_erases_record_support():
    np.testing.assert_array_equal(loss_events(2, 1, [0, 1]), [[True, False], [False, False]])
    np.testing.assert_array_equal(loss_events(2, 2, [0, 1]), [[True, True], [False, False]])
    np.testing.assert_array_equal(loss_events(0, 0, [0, 1]), [[False, False], [False, False]])
    with pytest.raises(ValueError):
        loss_events(1, 2, [0])
