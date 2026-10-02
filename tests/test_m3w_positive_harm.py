import inspect
import io
import numpy as np
import pytest
from scipy.optimize import minimize
from src.world_model import m3w_positive_harm as api
from test_m3w_past_quality_auxiliary import fixture


def sample():
    rng = np.random.default_rng(93); n = 100
    q = rng.normal(size=(n, 7)); y = np.column_stack((np.exp(.7*q[:, 0]), np.maximum(q[:, 1], 0)))
    return np.repeat([5, 8], n//2), q, y, rng.uniform(.1, 1, n)


def test_fit_preserves_training_mean_and_nonnegative_rates():
    leaves, q, y, w = sample(); fit = api.positive_leaf(leaves, q, y, w)
    for j, node in enumerate(fit['nodes']):
        mask = leaves == node; weight = w[mask]/w[mask].sum()
        rates = np.exp((q[mask]-fit['means'][j])@fit['coef'][j].T-fit['log_normalizer'][j])
        np.testing.assert_allclose(weight@rates, 1., atol=1e-9)
        assert (rates > 0).all()
    assert fit['maximum_gradient'] <= 1e-7
    assert all(v['after'] < v['before'] for v in fit['loss'])


def test_matches_independent_single_leaf_scipy_optimizer():
    _, q, y, w = sample(); leaves = np.zeros(len(w), int)
    a = api.positive_leaf(leaves, q, y, w); wn = w/w.sum(); z = q-wn@q
    for channel in range(2):
        target_mean = (wn*y[:, channel])@z/(wn@y[:, channel])
        def loss(b): return np.log(wn@np.exp(z@b))-target_mean@b+.5*b@b
        solved = minimize(loss, np.zeros(7), method='BFGS', options={'gtol': 1e-7})
        assert np.linalg.norm(solved.jac) < 1e-5
        np.testing.assert_allclose(a['coef'][0, channel], solved.x, atol=2e-6)


def test_zero_harm_leaves_stay_unchanged_without_pseudo_labels():
    leaves, q, y, w = sample(); y[:] = 0
    a = api.positive_leaf(leaves, q, y, w)
    np.testing.assert_array_equal(a['coef'], 0)
    np.testing.assert_array_equal(a['log_normalizer'], 0)
    assert a['zero_target'].all() and all(v['before'] == v['after'] == 0 for v in a['loss'])


def test_nonconvergence_is_not_silently_accepted():
    leaves, q, y, w = sample()
    with pytest.raises(RuntimeError, match='not converged'):
        api.positive_leaf(leaves, q, y, w, max_iter=1, tolerance=1e-12)


def test_inference_signature_excludes_future_and_targets():
    assert list(inspect.signature(api.predict).parameters) == ['state', 'fitted', 'x', 'envelope', 'past_quality']


def test_negative_training_harm_rejected():
    leaves, q, y, w = sample(); y[0, 0] = -1
    with pytest.raises(ValueError): api.positive_leaf(leaves, q, y, w)


def test_training_reproducible():
    args = sample(); a = api.positive_leaf(*args); b = api.positive_leaf(*args)
    for k in a:
        if isinstance(a[k], np.ndarray): np.testing.assert_array_equal(a[k], b[k])
        else: assert a[k] == b[k]


def test_forest_fit_replay_serialization_and_frozen_raw_moments():
    state, x, e, y, q, s, r, f = fixture()
    a = api.fit(state, x, e, y, q, s, r, f, settings={})
    b = api.fit(state, x, e, y, q, s, r, f, settings={})
    for k in a:
        if isinstance(a[k], np.ndarray): np.testing.assert_array_equal(a[k], b[k])
        else: assert a[k] == b[k]
    out = api.predict(state, a, x, e, q)
    np.testing.assert_array_equal(out[0], api.base.forest.predict(state, x, e)[0])
    stream = io.BytesIO()
    np.savez_compressed(stream, **{k: v for k, v in a.items() if isinstance(v, np.ndarray)})
    with np.load(io.BytesIO(stream.getvalue()), allow_pickle=False) as z:
        loaded = {k: z[k] for k in z.files}
    again = api.predict(state, loaded, x, e, q)
    for u, v in zip(out[:3], again[:3]): np.testing.assert_array_equal(u, v)
    assert out[3] == again[3] and a['unknown_train_rows'] == 1
    assert out[3]['exponential_guard_coordinates'] == 0


def test_zero_quality_is_exact_original_control():
    state, x, e, y, q, s, r, f = fixture(); q[:] = 0
    a = api.fit(state, x, e, y, q, s, r, f, settings={})
    old, new, _, _ = api.predict(state, a, x, e, q)
    np.testing.assert_array_equal(a['coef'], 0)
    np.testing.assert_array_equal(old, new)


def test_unknown_features_cannot_fit_normalization_or_harm():
    state, x, e, y, q, s, r, f = fixture()
    a = api.fit(state, x, e, y, q, s, r, f, settings={}); q[-1] = 1e8
    b = api.fit(state, x, e, y, q, s, r, f, settings={})
    for k in ('coef', 'means', 'log_normalizer', 'quality_mean', 'quality_std'):
        np.testing.assert_array_equal(a[k], b[k])


def test_modified_training_target_is_rejected():
    state, x, e, y, q, s, r, f = fixture(); y[0, 1] += 1
    with pytest.raises(AssertionError): api.fit(state, x, e, y, q, s, r, f, settings={})


def test_three_arm_readout_preserves_query_matching():
    state, x, e, y, q, s, r, f = fixture()
    fit = api.fit(state, x, e, y, q, s, r, f, settings={})
    old, new, support, _ = api.predict(state, fit, x, e, q)
    result, actions = api.evaluate(state, dict(original=old, positive=new, additive=old),
        y, e, np.ones(len(x), bool), support, s, r, f//4, np.arange(len(x)))
    assert 'positive' in result['scores'] and 'placebo' not in result['scores']
    for other in ('original', 'additive'):
        for frame in np.unique(f//4):
            assert actions[other+'_matched_positive'][f//4 == frame].sum() == actions['positive_matched_'+other][f//4 == frame].sum()
    assert result['policies']['positive']['unknown_rows'] == 1
