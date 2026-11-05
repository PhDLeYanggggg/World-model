import numpy as np
import pytest
from src.world_model import m3w_candidate_neighbor_basis as api
from src.evaluation import m3w_candidate_neighbor_probe as probe
from tests.test_m3w_rollout_history_risk import geometry


def packet():
    x = geometry()
    h = x[..., :24].reshape(2, 7, 8, 3); h[..., 2] = np.arange(-7, 1)/8
    n = x[..., 32:224].reshape(2, 7, 8, 8, 3)
    n[..., 0, :, 0] = 6; n[..., 0, :, 1] = 0
    n[..., 0, :, 2] = h[..., 2]
    x[..., 224:288].reshape(2, 7, 8, 8)[..., 0, :] = 1
    return x


def test_known_paths_and_static_neighbors():
    x = packet(); y = api.basis(x)
    assert y.shape == (2, 7, len(api.NAMES)) == (2, 7, 24)
    np.testing.assert_allclose(y[0, 0, :4], [1/8, 0, 6, 6])
    assert y[0, 0, 4] == 0 and y[0, 0, 9] == .5
    assert y[0, 0, 12] == pytest.approx(37**.5)
    assert y[0, 0, 20] == pytest.approx(37**.5)
    np.testing.assert_array_equal(api.chunked(x, 1), y)


def test_no_neighbors_masked_values_and_unaligned_time():
    x = packet(); x[..., 224:288] = 0
    np.testing.assert_array_equal(api.basis(x), np.zeros((2, 7, 24)))
    x[..., 32:224] = 1e10
    np.testing.assert_array_equal(api.basis(x), np.zeros((2, 7, 24)))
    x = packet(); x[..., 32:224].reshape(2, 7, 8, 8, 3)[..., 0, -1, 2] = 99
    np.testing.assert_array_equal(api.basis(x), np.zeros((2, 7, 24)))


def test_rigid_transform_and_neighbor_order_invariance():
    x = packet(); before = api.basis(x); rotation = np.array([[0., 1.], [-1., 0.]])
    for start, stop, shape in ((0, 24, (2, 7, 8, 3)), (32, 224, (2, 7, 8, 8, 3))):
        h = x[..., start:stop].reshape(shape); h[..., :2] = h[..., :2]@rotation+35
    for start in (306, 330):
        p = x[..., start:start+24].reshape(2, 7, 12, 2); p[:] = p@rotation
    n = x[..., 32:224].reshape(2, 7, 8, 8, 3); n[:] = n[..., ::-1, :, :].copy()
    m = x[..., 224:288].reshape(2, 7, 8, 8); m[:] = m[..., ::-1, :].copy()
    np.testing.assert_allclose(api.basis(x), before, atol=1e-6)


def test_feature_api_rejects_wrong_schema_and_ignores_unrelated_values():
    x = packet(); before = api.basis(x); x[..., 354:] = 9999
    np.testing.assert_array_equal(api.basis(x), before)
    with pytest.raises(ValueError):
        api.basis(x[..., :-1])
    x[..., 224] = .5
    with pytest.raises(ValueError):
        api.basis(x)


def test_probe_preprocessing_and_fit_do_not_use_held_values_or_labels():
    rng = np.random.default_rng(17); x = rng.normal(size=(60, 4)); y = rng.normal(size=(60, 2))
    train = np.arange(60) < 40; held = ~train; w = np.ones(60)
    a, sa = probe.fit_predict(x, y, train, held, w)
    xx, yy = x.copy(), y.copy(); xx[held] *= 100; yy[held] = 1e9
    _, sb = probe.fit_predict(xx, yy, train, held, w)
    for key in sa:
        np.testing.assert_array_equal(sa[key], sb[key])
    b, _ = probe.fit_predict(x, yy, train, held, w)
    np.testing.assert_array_equal(a, b)


def test_query_balance_and_original_sum_cost():
    sites = np.array(['a', 'a', 'a', 'b']); query = np.array([0, 0, 1, 2])
    np.testing.assert_allclose(probe.balanced_weights(sites, query), [.125, .125, .25, .5])
    result = probe.metrics(np.array([[-1., -1.]]), np.array([[2., 3.]]), np.ones(1), np.ones(2))
    assert result['total_joint_cost'] == 5


def test_probe_replay_and_synthetic_predictive_control():
    rng = np.random.default_rng(9); sites = np.repeat(['a', 'b', 'c'], 100)
    common = rng.normal(size=(300, 4)); added = rng.normal(size=(300, 24))
    y = np.column_stack((added[:, 0], added[:, 1])); query = np.arange(300)
    a, arrays = probe.probe(common, added, y, sites, query)
    b, again = probe.probe(common, added, y, sites, query)
    assert a == b and a['relative_MSE_improvement'] > .9
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], again[key])
