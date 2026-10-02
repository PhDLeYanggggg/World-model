import numpy as np
from src.world_model import m3w_positive_harm_diagnostic_v2 as api
from test_m3w_past_quality_auxiliary import fixture


def test_legacy_float32_label_transform_and_sequential_normalization():
    y = np.array([[.123456789, 2.3456789, 3.456789, 4.567891, 5.678912]], np.float32)
    a = np.array([[.5, 1.2, 2.5, 1.5, .8]]); b = a.copy(); b[:, [1, 4]] *= 1.7
    scale = 1.3456789123; rms = np.array([.123456789, .234567891, .345678912], np.float32)
    d = api.attribution(a, b, y, scale, rms)
    signed = api.forest.core.signed
    old = np.mean(((signed(a)-signed(y))/scale/rms)**2, 1)
    new = np.mean(((signed(b)-signed(y))/scale/rms)**2, 1)
    np.testing.assert_array_equal(d['old_MSE'], old)
    np.testing.assert_array_equal(d['new_MSE'], new)
    np.testing.assert_allclose(d['moments'].sum(1), new-old, atol=1e-12, rtol=1e-12)
    prior = api.v1.attribution(a, b, y, scale*rms)
    assert abs(float((prior['new_MSE']-prior['old_MSE']-(new-old))[0])) > 1e-7


def test_weighted_score_equals_frozen_reader_with_original_target_dtype():
    state, x, e, y, q, s, r, f = fixture(); y = y.astype(np.float32)
    rng = np.random.default_rng(14); a = rng.uniform(size=y.shape); b = a.copy()
    b[:, (1, 4)] *= 1.3
    w, _ = api.forest.core.weights(s, r, f, np.isfinite(y).all(1))
    d = api.error_slice(a, b, y, w, state['preprocess'], np.ones(len(y), bool))
    for name, pred in [('old', a), ('new', b)]:
        np.testing.assert_allclose(d['global_weighted_'+name+'_MSE'],
            api.forest.signed_score(pred, y, state['preprocess'], s, r, f), atol=1e-12, rtol=1e-12)
    assert d['unknown_rows'] == 1
    np.testing.assert_array_equal(np.array(d['moment_contributions'])[[0, 2, 3]], 0)
    empty = api.error_slice(a, b, y, w, state['preprocess'], np.isnan(y).all(1))
    assert empty['conditional_MSE_change'] is None
