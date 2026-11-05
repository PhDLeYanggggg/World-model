import numpy as np
import pytest
from src.evaluation import m3w_channel_specialist as api


def sample():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(260, 4)); y = np.column_stack((x[:, 0], np.maximum(x[:, 1], 0)))
    return x, y, np.arange(len(x)) < 220, np.ones(len(x))


def test_source_held_labels_cannot_affect_training_or_scalers():
    x, y, fit, w = sample()
    a = api.fit(x, y, fit, w, 'separate64')
    changed = y.copy(); changed[~fit] = 1e8
    b = api.fit(x, changed, fit, w, 'separate64')
    for k, v in api.fingerprint(a).items(): np.testing.assert_array_equal(v, api.fingerprint(b)[k])
    np.testing.assert_array_equal(api.predict(a, x[~fit]), api.predict(b, x[~fit]))
    assert a['trees'] == 64


def test_channels_have_no_cross_target_fitting_dependence():
    x, y, fit, w = sample()
    a = api.fit(x, y, fit, w, 'separate64')
    changed = y.copy(); changed[:, 0] = 100*y[:, 0]+50
    b = api.fit(x, changed, fit, w, 'separate64')
    np.testing.assert_array_equal(api.predict(a, x)[:, 1], api.predict(b, x)[:, 1])


def test_shared32_is_exact_prefix_of_existing_control():
    x, y, fit, w = sample()
    _, control = api.parent.fit_predict(x, y, fit, ~fit, w, 'extra_trees')
    control['model'].estimators_ = control['model'].estimators_[:32]
    a = api.fit(x, y, fit, w, 'shared32')
    np.testing.assert_array_equal(api.predict(a, x), api.parent.predict(control, x))
    np.testing.assert_array_equal(a['target_scale'], control['target_scale'])


def test_invalid_features_or_empty_training_fail():
    x, y, fit, w = sample()
    with pytest.raises(ValueError): api.fit(x, y, np.zeros_like(fit), w, 'separate64')
    a = api.fit(x, y, fit, w, 'separate64')
    with pytest.raises(ValueError): api.predict(a, y)
