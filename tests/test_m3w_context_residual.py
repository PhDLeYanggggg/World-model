import copy
import numpy as np
import pytest
from src.world_model import m3w_context_residual as m


def fixture():
    rng = np.random.default_rng(12); n = 120
    x = rng.normal(size=(n, len(m.NAMES))); p = np.tile([3., 2., 1., .5], (n, 1))
    y = p.copy(); y[:, 3] += .2*(x[:, 0] > 0)
    return x, p, y, np.ones(n)/n, np.repeat(['a', 'b', 'c'], n//3)


def test_context_affine_scale_rotation_invariance_and_missing_neighbor():
    g = np.zeros((2, 476)); h = np.stack((np.arange(-7, 1), np.arange(-7, 1)**2), 1)
    g[:, :16] = h.ravel(); w = np.array([2., 4.]); r = np.ones((2, 12, 2)); p = r*2
    a = m.features(g, w, r, p)
    q = np.array([[0., -1.], [1., 0.]])
    gg = g.copy(); gg[:, :16] = (h@q*3).ravel()
    b = m.features(gg, w*3, r@q*3, p@q*3)
    np.testing.assert_allclose(a, b, equal_nan=True)
    assert np.isnan(a[:, 4:6]).all() and (a[:, 3] == 0).all()


def test_present_neighbor_and_no_future_channels():
    g = np.zeros((1, 476)); g[0, :16] = np.column_stack((np.arange(-7, 1), np.zeros(8))).ravel()
    g[0, 38:54] = np.column_stack((np.arange(-7, 1)+2, np.zeros(8))).ravel()
    g[0, 230:238] = 1; r = np.zeros((1, 12, 2))
    x = m.features(g, np.ones(1), r, r)
    assert x[0, 3] == 1 and x[0, 4] == pytest.approx(2/7) and x[0, 5] == 0
    g[0, 308:] = 1e9
    np.testing.assert_allclose(x, m.features(g, np.ones(1), r, r))


def test_fit_excludes_held_unknown_and_preserves_nested_costs():
    x, p, y, w, sites = fixture(); y[-1] = np.nan; w[-1] = 0
    fitted = m.fit(x, p, y, w, sites, 'd'); old = copy.deepcopy(fitted)
    for arm in m.ARMS:
        z = m.predict(fitted[arm], np.ones((7, len(m.NAMES)))*1e8, p[:7])
        np.testing.assert_array_equal(z[:, :3], p[:7, :3]); assert (z[:, 3] <= z[:, 1]).all()
        assert (z[:, 3] >= 0).all()
    assert old == fitted
    with pytest.raises(ValueError): m.fit(x, p, y, w, sites, 'a')
    w[-1] = 1
    with pytest.raises(ValueError): m.fit(x, p, y, w, sites, 'd')


def test_global_bias_normal_equations_and_deterministic_context_fit():
    x, p, y, w, sites = fixture(); a = m.fit(x, p, y, w, sites, 'd')
    assert a == m.fit(x.copy(), p.copy(), y.copy(), w.copy(), sites.copy(), 'd')
    z = m.predict(a['global_bias'], x, p)
    np.testing.assert_allclose(z[:, 3]-p[:, 3], np.mean(y[:, 3]-p[:, 3]))
    assert a['context_bias']['weighted_fitting_residual_MSE'] < a['global_bias']['weighted_fitting_residual_MSE']


def test_unknown_contexts_explicit_and_cut_uses_fitting_only():
    x, p, y, w, sites = fixture(); x[:, 5] = np.nan
    a = m.fit(x, p, y, w, sites, 'd'); assert a['context_bias']['cuts'][5] is None
    b = m.bins(np.ones((2, len(m.NAMES))), a['context_bias']['cuts']); assert (b[:, 5] == 3).all()
    xx = x.copy(); xx[-1, 0] = 1e10; y[-1] = np.nan; w[-1] = 0
    assert m.fit(xx, p, y, w, sites, 'd') == m.fit(x, p, y, w, sites, 'd')


def test_context_sign_uses_all_three_sites_and_preserves_unsupported_cells():
    def row(site, sign):
        return dict(site=site, cells=[[dict(rows=30, tracks=15, centered_bias=sign),
             *[dict(rows=0, tracks=0, centered_bias=None) for _ in range(3)]] for _ in m.NAMES])
    fit = [row(s, 1) for s in 'abc']; held = [row('d', -1)]
    r = m.repeated_contexts(fit, held)
    assert all(v['supported_cells'] == v['repeated_fitting_sign'] == v['held_opposite_sign'] == 1 for v in r)
    fit[1] = row('b', -1)
    assert all(v['repeated_fitting_sign'] == 0 for v in m.repeated_contexts(fit, held))
