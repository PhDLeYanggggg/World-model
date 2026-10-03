import inspect
import numpy as np
from src.world_model import m3w_cost_support_diagnostic as api
from test_m3w_past_quality_auxiliary import fixture


def fitted_fixture():
    args = fixture()
    state, x, e, y, q, s, r, f = args
    fitted = api.model.fit(*args, settings={})
    tables = api.training_support(state, fitted, x, e, y, q, s, r, f)
    return args, fitted, tables


def test_ranges_are_leaf_local_and_finite():
    q = np.arange(28).reshape(4, 7)
    table = api.quality_ranges(np.array([1, 1, 3, 3]), q)
    np.testing.assert_array_equal(table['lower'], q[[0, 2]])
    np.testing.assert_array_equal(table['upper'], q[[1, 3]])


def test_raw_inference_matches_frozen_cost_model_and_excludes_future():
    (state, x, e, y, q, s, r, f), fitted, tables = fitted_fixture()
    old, new, support, desc, extra = api.raw_predictions(state, fitted, x, e, q, tables)
    expected = api.model.predict(state, fitted, x, e, q)
    for a, b in ((api.forest.project_moments(old, e), expected[0]),
                 (api.forest.project_moments(new, e), expected[1]), (support, expected[2])):
        np.testing.assert_array_equal(a, b)
    assert list(inspect.signature(api.raw_predictions).parameters) == ['state', 'fitted', 'x', 'envelope', 'quality', 'tables']
    np.testing.assert_array_equal(extra[np.isfinite(y).all(1), 2], 0)
    assert desc.shape == (len(x), 6) and extra.shape == (len(x), 3)


def test_unknown_label_rows_do_not_define_training_support():
    (state, x, e, y, q, s, r, f), fitted, tables = fitted_fixture()
    q[~np.isfinite(y).all(1)] = 1e8
    again = api.training_support(state, fitted, x, e, y, q, s, r, f)
    for a, b in zip(tables, again):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_tree_ensemble_identity_and_training_surrogate_match():
    (state, x, e, y, q, s, r, f), fitted, tables = fitted_fixture()
    d = api.tree_score_diagnostic(state, fitted, x, e, q, y, s, r, f)
    np.testing.assert_allclose(d['mean_tree_MSE_change']-d['dispersion_change'],
                               d['raw_ensemble_MSE_change'], atol=1e-9)
    loss = fitted['training_loss']
    np.testing.assert_allclose(d['mean_tree_MSE_change']+d['fixed_penalty'],
                               sum(loss['after'])-sum(loss['before']), atol=1e-6, rtol=1e-6)
    old, new, _, _, _ = api.raw_predictions(state, fitted, x, e, q, tables)
    known = np.isfinite(y).all(1)
    w, _ = api.forest.core.weights(s, r, f, known)
    read = api.prior.error_slice(old, new, y, w, state['preprocess'], np.ones(len(y), bool))
    np.testing.assert_allclose(read['global_weighted_MSE_change'], d['raw_ensemble_MSE_change'], atol=1e-9)
    assert d['unknown_rows'] == 1


def test_better_tree_loss_need_not_mean_better_ensemble():
    old = np.array([[-1.], [1.]])
    new = np.array([[.6], [.6]])
    assert np.mean(new**2) < np.mean(old**2)
    assert np.mean(new, 0)**2 > np.mean(old, 0)**2


def test_empty_selected_support_is_undefined_not_zero():
    (state, x, e, y, q, s, r, f), fitted, tables = fitted_fixture()
    old, new, _, desc, extra = api.raw_predictions(state, fitted, x, e, q, tables)
    actions = {k: np.zeros(len(y), bool) for k in ('original', 'cost', 'original_matched_cost', 'cost_matched_original')}
    raw = dict(original=old, cost=new)
    projected = {k: api.forest.project_moments(v, e) for k, v in raw.items()}
    d = api.diagnose(state, raw, projected, y, actions, s, r, f, desc, extra)
    assert d['cohorts']['cost_selected']['conditional_MSE_change'] is None
    assert d['support_cohorts']['cost_selected']['train_effective_harm_mean'] is None
