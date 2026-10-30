import numpy as np
import pytest
from src.world_model import m3w_position_anchor as api
from src.world_model.m3w_european_source_forecast import baseline_numpy


@pytest.mark.parametrize('window,index', [(4, 4), (8, 5)])
def test_raw_control_is_bit_exact_and_slope_is_unchanged(window, index):
    h = np.random.default_rng(7).normal(size=(12, 8, 2))
    raw = api.predict(h, window=window, anchor='raw_last')
    np.testing.assert_array_equal(raw, baseline_numpy(h, index))
    fitted = api.predict(h, window=window, anchor='causal_fitted')
    shift = fitted-raw
    np.testing.assert_allclose(shift, np.broadcast_to(shift[:, :1], shift.shape), atol=1e-14)
    np.testing.assert_allclose(np.diff(fitted, axis=1), np.diff(raw, axis=1), atol=1e-14)


def test_exact_linear_motion_needs_no_position_repair():
    h = np.stack((np.arange(8)*2+3, np.arange(8)*-1+2), 1)[None].astype(float)
    for k in (4, 8):
        np.testing.assert_array_equal(api.predict(h, window=k, anchor='raw_last'),
            api.predict(h, window=k, anchor='causal_fitted'))


def test_fitted_position_is_past_only_and_translation_equivariant():
    rng = np.random.default_rng(8); h = rng.normal(size=(5, 8, 2)); offset = np.array([11., -9.])
    for k in (4, 8):
        a = api.predict(h, window=k, anchor='causal_fitted')
        b = api.predict(h+offset, window=k, anchor='causal_fitted')
        np.testing.assert_allclose(b, a+offset, atol=1e-13)
        np.testing.assert_allclose(api.predict(3*h, window=k, anchor='causal_fitted'), 3*a, atol=1e-13)
    with pytest.raises(TypeError): api.predict(h, window=8, anchor='causal_fitted', future=np.zeros((5, 12, 2)))


def test_noise_mechanism_is_synthetic_not_a_real_world_success():
    rng = np.random.default_rng(27); n = 3000
    clean = np.broadcast_to(np.arange(8)[None, :, None]*np.array([.5, -.2]), (n, 8, 2))
    h = clean+rng.normal(scale=.3, size=clean.shape)
    future = np.arange(8, 20)[None, :, None]*np.array([.5, -.2])
    raw = api.predict(h, window=8, anchor='raw_last'); fit = api.predict(h, window=8, anchor='causal_fitted')
    assert np.square(fit-future).mean() < np.square(raw-future).mean()


def test_invalid_or_nonfinite_history_refuses():
    with pytest.raises(ValueError): api.predict(np.zeros((2, 9, 2)), window=8, anchor='causal_fitted')
    with pytest.raises(ValueError): api.predict(np.zeros((2, 8, 2)), window=3, anchor='causal_fitted')
    with pytest.raises(ValueError): api.predict(np.full((2, 8, 2), np.nan), window=8, anchor='raw_last')
    with pytest.raises(ValueError): api.predict(np.zeros((2, 8, 2)), window=8, anchor='future_fit')
