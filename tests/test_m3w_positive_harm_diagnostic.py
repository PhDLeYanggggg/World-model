import inspect
import math
import numpy as np
from src.world_model import m3w_positive_harm_diagnostic as api
from src.world_model import m3w_positive_harm as trained
from test_m3w_past_quality_auxiliary import fixture


def test_quadratic_attribution_matches_independent_scalar_expansion():
    rng = np.random.default_rng(71)
    a, b, y = [rng.uniform(size=(23, 5)) for _ in range(3)]
    scale = np.array([.3, 2., 5.])
    d = api.attribution(a, b, y, scale)
    for i in range(len(y)):
        sa = [a[i, 0]-a[i, 1], a[i, 1]-.02*a[i, 2], a[i, 4]-.02*a[i, 3]]
        sb = [b[i, 0]-b[i, 1], b[i, 1]-.02*b[i, 2], b[i, 4]-.02*b[i, 3]]
        sy = [y[i, 0]-y[i, 1], y[i, 1]-.02*y[i, 2], y[i, 4]-.02*y[i, 3]]
        scalar = math.fsum(((sb[j]-sy[j])**2-(sa[j]-sy[j])**2)/scale[j]**2/3 for j in range(3))
        np.testing.assert_allclose(d['moments'][i].sum(), scalar, atol=1e-12)
        np.testing.assert_allclose(d['scores'][i].sum(), scalar, atol=1e-12)


def test_reference_contributions_are_exactly_zero_when_frozen():
    rng = np.random.default_rng(6); a = rng.uniform(size=(40, 5)); b = a.copy()
    b[:, (1, 4)] *= 2
    d = api.attribution(a, b, np.zeros_like(a), np.ones(3))
    np.testing.assert_array_equal(d['moments'][:, (0, 2, 3)], 0)


def test_weighted_effective_harm_support_scalar_reference():
    leaves = np.array([3, 3, 3, 8, 8]); weights = np.array([1., 2., 4., 2., 1.])
    harm = np.array([[1., 1.], [0., 0.], [1., 0.], [0., 0.], [0., 0.]])
    d = api.leaf_support(leaves, weights, harm)
    np.testing.assert_allclose(d['effective_harm'][0], [25/17, 1.])
    np.testing.assert_array_equal(d['effective_harm'][1], 0)
    np.testing.assert_array_equal(d['harm_count'], [[2, 1], [0, 0]])


def test_raw_prediction_matches_registered_model_and_has_no_future_arguments():
    state, x, e, y, q, s, r, f = fixture()
    fit = trained.fit(state, x, e, y, q, s, r, f, settings={})
    table = api.training_support(state, x, e, y, s, r, f)
    old, new, support, desc = api.raw_positive(state, fit, x, e, q, table)
    expected = trained.predict(state, fit, x, e, q)
    for actual, want in ((api.forest.project_moments(old, e), expected[0]),
                         (api.forest.project_moments(new, e), expected[1]), (support, expected[2])):
        np.testing.assert_array_equal(actual, want)
    assert desc.shape == (len(x), 6) and np.isfinite(desc).all()
    assert list(inspect.signature(api.raw_positive).parameters) == ['state', 'fitted', 'x', 'envelope', 'past_quality', 'train_support']


def test_cohort_reports_unknown_counts_without_fake_zero_error():
    state, x, e, y, q, s, r, f = fixture()
    fit = trained.fit(state, x, e, y, q, s, r, f, settings={})
    old, new, support = trained.predict(state, fit, x, e, q)[:3]
    report = api.error_slice(old, new, y, np.full(len(y), 1/len(y)),
                             state['preprocess'], np.ones(len(y), bool))
    assert report['unknown_rows'] == 1 and report['known_rows'] == len(y)-1
    empty = api.error_slice(old, new, y, np.full(len(y), 1/len(y)),
                            state['preprocess'], np.isnan(y).all(1))
    assert empty['known_rows'] == 0 and empty['conditional_MSE_change'] is None


def test_decomposition_is_additive_across_disjoint_slices():
    state, x, e, y, q, s, r, f = fixture()
    fit = trained.fit(state, x, e, y, q, s, r, f, settings={})
    old, new, _ = trained.predict(state, fit, x, e, q)[:3]
    w = np.full(len(y), 1/len(y)); mask = np.arange(len(y)) % 2 == 0
    parts = [api.error_slice(old, new, y, w, state['preprocess'], m)
             for m in (mask, ~mask, np.ones(len(y), bool))]
    assert math.isclose(parts[0]['global_weighted_MSE_change']+parts[1]['global_weighted_MSE_change'],
                        parts[2]['global_weighted_MSE_change'], abs_tol=1e-12)
