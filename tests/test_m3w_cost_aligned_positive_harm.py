import inspect
import numpy as np
import pytest
from scipy.optimize import minimize
from src.world_model import m3w_cost_aligned_positive_harm as api
from test_m3w_positive_harm import sample
from test_m3w_past_quality_auxiliary import fixture


def test_leaf_squared_cost_matches_independent_optimizer():
    _, q, y, w = sample(); leaves = np.zeros(len(w), int); scales = np.array([.8, 1.1])
    out = api.cost_leaf(leaves, q, y, w, scales)
    wn = w/w.sum(); z = q-wn@q; mu = wn@y
    for channel in range(2):
        def loss(beta):
            r = np.exp(z@beta); p = mu[channel]*r/(wn@r)
            return .5*(wn@((p-y[:, channel])/scales[channel])**2)+.5*(beta@beta)
        fit = minimize(loss, np.zeros(7), method='BFGS', options={'gtol': 1e-7})
        np.testing.assert_allclose(out['coef'][0, channel], fit.x, atol=3e-6)
        rate = np.exp(z@out['coef'][0, channel]-out['log_normalizer'][0, channel])
        np.testing.assert_allclose(wn@rate, 1., atol=1e-12)
    assert out['maximum_gradient'] <= 1e-7
    assert all(d['after'] < d['before'] for d in out['loss'])


def test_zero_harm_unchanged_and_nonconvergence_explicit():
    leaves, q, y, w = sample(); a = api.cost_leaf(leaves, q, np.zeros_like(y), w, np.ones(2))
    np.testing.assert_array_equal(a['coef'], 0)
    np.testing.assert_array_equal(a['log_normalizer'], 0)
    with pytest.raises(RuntimeError, match='not converged'):
        api.cost_leaf(leaves, q, y, w, np.ones(2), max_iter=1, tolerance=1e-12)


def test_same_positive_link_preserves_train_means_with_cost_sensitive_strength():
    leaves, q, y, w = sample()
    normal = api.cost_leaf(leaves, q, y, w, np.ones(2))
    tiny = api.cost_leaf(leaves, q, y*1e-4, w, np.ones(2))
    assert np.linalg.norm(tiny['coef']) < np.linalg.norm(normal['coef'])*1e-3
    for i, node in enumerate(normal['nodes']):
        at = leaves == node; wn = w[at]/w[at].sum()
        rates = np.exp((q[at]-normal['means'][i])@normal['coef'][i].T-normal['log_normalizer'][i])
        np.testing.assert_allclose(wn@rates, 1, atol=1e-12)
    assert 'target' not in inspect.signature(api.predict).parameters


def test_cost_change_equals_three_score_surrogate_when_leaf_means_preserved():
    _, q, y, w = sample(); rng = np.random.default_rng(52)
    five = rng.uniform(size=(len(w), 5)); five[:, (1, 4)] = y
    wn = w/w.sum(); old = np.tile(wn@five, (len(w), 1))
    sigma = np.array([.7, 1.3, .9]); scale = api.harm_scales(2., sigma)
    equivalent = api.score_equivalent_targets(five, old, sigma)
    fit = api.cost_leaf(np.zeros(len(w), int), q, y, w, scale, squared_target=equivalent)
    new = old.copy(); rate = np.exp((q-fit['means'][0])@fit['coef'][0].T-fit['log_normalizer'][0])
    new[:, (1, 4)] *= rate
    signed = api.positive.base.forest.core.signed
    difference = wn@(np.mean(((signed(new)-signed(five))/2./sigma)**2, 1)-np.mean(((signed(old)-signed(five))/2./sigma)**2, 1))
    half_moment_change = wn@(.5*np.sum(((new[:, (1, 4)]-equivalent)/scale)**2-((old[:, (1, 4)]-equivalent)/scale)**2, 1))
    np.testing.assert_allclose(difference, half_moment_change, atol=1e-12)


def test_forest_replay_unknown_exclusion_and_frozen_predictions():
    state, x, e, y, q, s, r, f = fixture()
    a = api.fit(state, x, e, y, q, s, r, f, settings={})
    b = api.fit(state, x, e, y, q, s, r, f, settings={})
    for k in a:
        if isinstance(a[k], np.ndarray): np.testing.assert_array_equal(a[k], b[k])
        else: assert a[k] == b[k]
    q2 = q.copy(); q2[-1] = 1e6
    c = api.fit(state, x, e, y, q2, s, r, f, settings={})
    for k in ('coef', 'means', 'log_normalizer', 'quality_mean', 'quality_std'):
        np.testing.assert_array_equal(a[k], c[k])
    old, new, support, _ = api.predict(state, a, x, e, q)
    np.testing.assert_array_equal(old, api.positive.base.forest.predict(state, x, e)[0])
    out, actions = api.evaluate(state, dict(original=old, additive=old, poisson=old, cost=new),
        y, e, np.ones(len(x), bool), support, s, r, f//4, np.arange(len(x)))
    assert out['policies']['cost']['unknown_rows'] == 1
    assert len(actions) == 10 and len(out['contrasts']) == 9


def test_invalid_scale_and_effective_targets_rejected():
    leaves, q, y, w = sample()
    with pytest.raises(ValueError): api.cost_leaf(leaves, q, y, w, np.array([0., 1.]))
    with pytest.raises(ValueError): api.cost_leaf(leaves, q, y, w, np.ones(2), squared_target=np.full_like(y, np.nan))
