import numpy as np

from src.world_model.m3w_exchange_amplitude_control import amplitude_control
from src.world_model.m3w_exchange_phase import phase_paths


def test_amplitude_control_matches_late_without_reordering_interior():
    cv = np.zeros((1, 12, 2))
    p = cv.copy()
    p[0, :, 0] = np.arange(1, 13)
    late = phase_paths(cv, p)['late']
    x, valid = amplitude_control(cv, p, late)
    assert valid.tolist() == [True]
    np.testing.assert_array_equal(x[:, -1], p[:, -1])
    np.testing.assert_allclose(np.linalg.norm(x-cv, axis=-1).sum(1), np.linalg.norm(late-cv, axis=-1).sum(1))
    np.testing.assert_allclose(x[0, :-1, 0]/p[0, :-1, 0], np.repeat(x[0, 0, 0], 11))
    assert not np.array_equal(x, late)


def test_impossible_amplitude_target_does_not_change_original():
    cv = np.zeros((1, 12, 2))
    p = cv.copy()
    p[0, -1, 0] = 1.
    target = p+1.
    x, valid = amplitude_control(cv, p, target)
    assert valid.tolist() == [False]
    np.testing.assert_array_equal(x, p)
