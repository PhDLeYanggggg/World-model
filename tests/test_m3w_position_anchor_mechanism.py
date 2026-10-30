import numpy as np
import pytest
from src.evaluation.m3w_position_anchor_mechanism import decompose
from src.world_model.m3w_position_anchor import predict


def test_exact_quadratic_motion_fitted_endpoint_lags_observed_endpoint():
    t = np.arange(-7, 1); acceleration = np.array([2., -1.])
    h = (.5*t[:, None]**2*acceleration)[None]
    for k in (4, 8):
        raw = predict(h, window=k, anchor='raw_last'); fitted = predict(h, window=k, anchor='causal_fitted')
        np.testing.assert_allclose(fitted[:, 0]-raw[:, 0], -acceleration[None]*(k-1)*(k-2)/12)


def test_squared_error_identity_and_unknown_rows_are_explicit():
    rng = np.random.default_rng(4); raw = rng.normal(size=(4, 12, 2)); fitted = raw+.2
    y = rng.normal(size=raw.shape); valid = np.ones((4, 12), bool); valid[1] = False; valid[2, 8:] = False
    y[1] = np.nan
    result = decompose(raw, fitted, y, valid, rng.normal(size=(4, 8, 2)), np.ones(4))
    a = result['terms']; assert a['direct'] == pytest.approx(a['cross']+a['squared_shift'], abs=1e-14)
    assert result['known_rows'] == 3 and result['unknown_rows'] == 1
    assert result['per_step'][9]['observed_rows'] == 2


def test_all_unknown_support_does_not_impute_zero_error():
    z = np.zeros((2, 12, 2)); h = np.zeros((2, 8, 2))
    r = decompose(z, z, np.full_like(z, np.nan), np.zeros((2, 12), bool), h, np.ones(2))
    assert all(v is None for v in r['terms'].values()) and r['max_abs_identity_error'] is None
    assert r['past_acceleration_alignment']['mean_scaled_dot'] is None
