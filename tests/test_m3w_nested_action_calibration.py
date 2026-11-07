import numpy as np
import pytest

from src.world_model import m3w_nested_action_calibration as m


def test_observed_components_keep_unknown_completion_and_no_reference():
    floor = np.zeros((3, 12, 2))
    path = np.ones((3, 1, 12, 2))
    target = floor.copy(); target[0] = .1; target[1] = 10.; target[2] = np.nan
    valid = np.ones((3, 12), bool); valid[-1] = False
    out = m.observed_components(path, floor, target, valid, np.full((3, 1), np.sqrt(2.)), .2)
    np.testing.assert_allclose(out[0, 0], np.sqrt(2.)*np.array([.8, .1, .8, .1]))
    np.testing.assert_allclose(out[1, 0], np.sqrt(2.)*np.array([0., 10., 0., 0.]))
    np.testing.assert_allclose(out[2, 0], np.sqrt(2.)*np.array([1., 0., 1., 0.]))


def test_calibrator_excludes_held_and_weights_source_not_rows():
    donors = {k: [dict(residual=[n, -n, 0, 0])] for n, k in enumerate('abc')}
    np.testing.assert_array_equal(m.calibrate(donors, 'd'), [2, 0, 0, 0])
    with pytest.raises(ValueError, match='Target source'):
        m.calibrate(donors, 'a')
    assert m.calibrate(dict(a=[]), 'c') is None


def test_numerator_and_reference_offsets_are_separate():
    z = np.zeros((2, 2)); radius = np.ones_like(z)
    margins = [1., 2., 3., 4.]
    a, e = m.adjusted_excess('joint', z, z, radius, margins)
    np.testing.assert_allclose(a, 1.04); np.testing.assert_allclose(e, 3.08)
    _, e = m.adjusted_excess('reference_only', z, z, radius, margins)
    np.testing.assert_allclose(e, .08)
