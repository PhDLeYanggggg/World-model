import numpy as np

from src.evaluation.m3w_damped_motion_options import fixed_recipient_paths, options


def test_rotation_translation_equivariance_and_speed_matching():
    theta = np.arange(7)*.22
    history = np.r_[np.zeros((1, 2)), np.cumsum(np.column_stack((np.cos(theta), np.sin(theta))), axis=0)][None]
    ols = np.arange(1, 13)[None, :, None]*np.array([[[1., .3]]])
    cv = np.zeros_like(ols)
    s, t, _, signal = options(history, ols, cv)
    assert signal.tolist() == [True]
    delta = lambda p: np.diff(p, axis=2, prepend=np.zeros_like(p[:, :, :1]))
    np.testing.assert_allclose(np.linalg.norm(delta(s), axis=-1), np.linalg.norm(delta(t), axis=-1))
    r = np.array([[0., -1.], [1., 0.]])
    ss, tt, _, _ = options(history @ r + 100., ols @ r, cv @ r)
    np.testing.assert_allclose(ss, s @ r, atol=1e-9)
    np.testing.assert_allclose(tt, t @ r, atol=1e-9)


def test_fixed_switch_recipients_even_for_identity_option():
    cv = np.zeros((3, 12, 2))
    original = cv.copy()
    original[:2] = 1.
    straight, turn = cv.copy(), cv.copy()
    straight[[0, 2]] = 2.
    turn[[1, 2]] = 3.
    s, t = fixed_recipient_paths(original, cv, straight, turn, np.ones(3, bool), np.ones(3, bool))
    np.testing.assert_array_equal((s != cv).any(axis=(1, 2)), [True, True, False])
    np.testing.assert_array_equal((t != cv).any(axis=(1, 2)), [True, True, False])
    np.testing.assert_array_equal(s[1], original[1])
