import inspect
import numpy as np
import pytest
from src.world_model import m3w_leaf_quality_extension as api
from test_m3w_cost_support_diagnostic import fitted_fixture


def case():
    args, fitted, _ = fitted_fixture()
    return args, fitted, api.training_bounds(args[0], fitted, *args[1:])


def test_known_training_predictions_are_exactly_unchanged():
    (s, x, e, y, q, sites, recordings, frames), fitted, bounds = case()
    raw, supported, changed = api.predict_raw(s, fitted, bounds, x, e, q)
    known = np.isfinite(y).all(1)
    np.testing.assert_array_equal(raw['cost'][known], raw['extended'][known])
    np.testing.assert_array_equal(changed[known], 0)
    old, cost, sup, _ = api.diagnostic.model.predict(s, fitted, x, e, q)
    for a, b in ((old, api.forest.project_moments(raw['original'], e)),
                 (cost, api.forest.project_moments(raw['cost'], e)), (supported, sup)):
        np.testing.assert_array_equal(a, b)
    assert list(inspect.signature(api.predict_raw).parameters) == ['state', 'fitted', 'bounds', 'x', 'envelope', 'quality']


def test_unknown_training_quality_does_not_define_bounds():
    (s, x, e, y, q, sites, recordings, frames), fitted, bounds = case()
    q[~np.isfinite(y).all(1)] = 1e6
    again = api.training_bounds(s, fitted, x, e, y, q, sites, recordings, frames)
    for key in bounds:
        np.testing.assert_array_equal(bounds[key], again[key])


def test_outside_quality_changes_harm_without_changing_reference_or_benefit():
    (s, x, e, y, q, sites, recordings, frames), fitted, bounds = case()
    q[:] = fitted['quality_mean']+8*fitted['quality_std']
    raw, _, changed = api.predict_raw(s, fitted, bounds, x, e, q)
    assert (changed > 0).any()
    assert not np.array_equal(raw['cost'], raw['extended'])
    for name in ('cost', 'extended'):
        np.testing.assert_array_equal(raw[name][:, (0, 2, 3)], raw['original'][:, (0, 2, 3)])
        assert np.isfinite(raw[name]).all() and (raw[name] >= 0).all()
    again = api.predict_raw(s, fitted, bounds, x, e, q)[0]
    np.testing.assert_array_equal(raw['extended'], again['extended'])


def test_clip_identity_boundary_and_constant_dimensions():
    q = np.tile(np.arange(7.), (3, 1))
    lo, hi = np.zeros_like(q), np.full_like(q, 3.)
    lo[:, 1] = hi[:, 1] = 2
    expected = np.tile([0., 2., 2., 3., 3., 3., 3.], (3, 1))
    np.testing.assert_array_equal(api.clip_quality(q, lo, hi), expected)
    np.testing.assert_array_equal(api.clip_quality(expected, lo, hi), expected)


@pytest.mark.parametrize('bad', ['nan', 'reversed', 'dimensions'])
def test_invalid_bounds_fail_closed(bad):
    q = np.ones((2, 7)); lo = np.zeros_like(q); hi = np.ones_like(q)
    if bad == 'nan': lo[0, 0] = np.nan
    if bad == 'reversed': lo[0, 0] = 2
    if bad == 'dimensions': hi = hi[:, :6]
    with pytest.raises(ValueError): api.clip_quality(q, lo, hi)


def test_unknown_labels_remain_in_policy_readout():
    s, x, e, y, q, sites, recordings, frames = fitted_fixture()[0]
    pred = np.tile([1., 0., 10., 10., 0.], (len(y), 1))
    predictions = {k: pred for k in api.CONTROLS+('extended',)}
    result, actions = api.evaluate(s, predictions, y, e, np.ones(len(y), bool), np.ones(len(y), bool),
                                   sites, recordings, frames, np.arange(len(y)))
    assert result['policies']['extended']['selected_unknown'] == 1
    assert actions['extended'][~np.isfinite(y).all(1)].all()
