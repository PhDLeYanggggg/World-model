import numpy as np
import pytest
from src.world_model.m3w_causal_motion_support import motion_profile, veto_eligibility


def example():
    t = np.tile(np.arange(-7, 1)*12, (3, 1))
    xy = t[..., None]*np.array([[[.5, -.25]], [[0., 0.]], [[1., 1.]]])
    xy[2, 2, 0] += .5
    return xy, t


def test_exact_native_cv_and_stationary_are_distinct():
    x, t = example(); p = motion_profile(x, t)
    np.testing.assert_array_equal(p['exact_past_cv'], [True, True, False])
    np.testing.assert_array_equal(p['stationary'], [False, True, False])
    np.testing.assert_array_equal(p['last_stop'], [False, True, False])
    assert p['max_backcast_error'][2] == .5


def test_stop_after_nonconstant_history_is_not_full_cv():
    x, t = example(); x[0, -1] = x[0, -2]
    p = motion_profile(x, t)
    assert p['last_stop'][0] and not p['exact_past_cv'][0]


def test_translation_rotation_and_integer_scale_preserve_exact_cases():
    x, t = example(); expected = motion_profile(x, t)['exact_past_cv']
    rotated = x[..., ::-1] * [1, -1]
    np.testing.assert_array_equal(motion_profile(rotated*4+[300, 900], t)['exact_past_cv'], expected)


def test_irregular_past_time_uses_velocity_not_step_distance():
    t = np.array([[-90, -78, -60, -45, -30, -20, -10, 0]])
    x = t[..., None]*np.array([[[.5, -.5]]])
    assert motion_profile(x, t)['exact_past_cv'][0]


@pytest.mark.parametrize('bad', ['future', 'duplicate', 'short', 'nan'])
def test_rejects_invalid_past_contract(bad):
    x, t = example()
    if bad == 'future': t[0, -1] = 1
    if bad == 'duplicate': t[0, 3] = t[0, 2]
    if bad == 'short': x = x[:, 1:]
    if bad == 'nan': x[0, 0, 0] = np.nan
    with pytest.raises(ValueError): motion_profile(x, t)


def test_veto_only_removes_eligible_rows():
    np.testing.assert_array_equal(veto_eligibility(np.array([True, False, True]),
        np.array([True, False, False])), [False, False, True])
    with pytest.raises(ValueError): veto_eligibility([1, 0], [True, False])


def test_annotation_equality_has_no_fitted_tolerance():
    x, t = example(); x[0, 2, 0] += 1e-10
    with pytest.raises(ValueError): motion_profile(x, t)


@pytest.mark.parametrize('kind', ['coordinate_range', 'frame_range', 'fractional_frame'])
def test_rejects_unsupported_exact_arithmetic_contract(kind):
    x, t = example(); t = t.astype(float)
    if kind == 'coordinate_range': x += 2**21
    if kind == 'frame_range': t *= 2**20
    if kind == 'fractional_frame': t[0, 0] += .5
    with pytest.raises(ValueError): motion_profile(x, t)
