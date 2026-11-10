import numpy as np
import pytest

from scripts.audit_m3w_opendd_angular_calibration_mask import selection_weights


def test_random_expectation_preserves_each_recordings_rotation_count():
    rec = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'c'])
    eligible = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)
    rotated = np.array([1, 0, 0, 0, 1, 1, 0], dtype=bool)
    weights, flexible = selection_weights(rec, eligible, rotated)
    np.testing.assert_allclose(weights, [1/3, 1/3, 1/3, 0, 1, 1, 0])
    assert flexible == 1
    rotated[3] = True
    with pytest.raises(ValueError, match='incumbent edits'):
        selection_weights(rec, eligible, rotated)
