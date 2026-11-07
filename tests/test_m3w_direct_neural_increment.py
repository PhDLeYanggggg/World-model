import numpy as np
import pytest

from scripts.run_m3w_direct_neural_increment import (
    LEARNED, fitting_sources, lower_target, replace_actions,
)


def test_outer_and_inner_sources_are_excluded_from_donor_anchors():
    assert fitting_sources(['a', 'b', 'c', 'd'], 'a', 'b') == ['c', 'd']
    with pytest.raises(ValueError, match='cannot be an anchor donor'):
        fitting_sources(['a', 'b', 'c'], 'a', 'a')


def test_direct_target_bounds_only_wholly_missing_rows():
    anchor = np.zeros((3, 2, 2))
    candidate = anchor.copy()
    candidate[..., 0] = [1., 5.]
    y = candidate.copy()
    valid = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
    np.testing.assert_allclose(lower_target(anchor, candidate[:, None], y, valid)[:, 0], [3., 1., -5.])


def test_replacement_preserves_recipients_and_rejects_ineligible_paths():
    anchor = np.array([-1, 3, 4, 5])
    legal = np.ones((4, 3), dtype=bool)
    legal[2, 1] = False
    gain = np.array([[2, 3, 4], [0, .1, -.1], [-1, 10, 0], [0, 0, 0.]])
    np.testing.assert_array_equal(replace_actions(anchor, legal, gain), [-1, LEARNED[1], 4, 5])
