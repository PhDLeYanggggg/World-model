import inspect
import numpy as np
import pytest
from src.world_model import m3w_temporal_support as m


def geometry():
    g = np.zeros((4, 476)); h = np.zeros((4, 8, 2))
    h[:, :, 0] = np.arange(-7, 1); h[1, :, 1] = np.arange(-7, 1)**2/10
    g[:, :16] = h.reshape(4, -1); g[:, 16:24] = np.arange(-7, 1)/12
    pos = g[:, 38:166].reshape(4, 8, 8, 2); pos[:, 0] = h+[4, 2]
    g[:, 166:230].reshape(4, 8, 8)[:, 0] = np.arange(-7, 1)/12
    g[:, 230:294].reshape(4, 8, 8)[:, 0] = 1
    return g


def test_past_feature_shapes_padding_and_rotation():
    g = geometry(); a, b = m.temporal_features(g, np.ones(4))
    assert a.shape == b.shape == (4, 68)
    changed = g.copy(); changed[:, 308:] = 9e5
    changed[:, 38:166].reshape(4, 8, 8, 2)[:, 1:] = 7e6
    aa, bb = m.temporal_features(changed, np.ones(4))
    np.testing.assert_array_equal(a, aa); np.testing.assert_array_equal(b, bb)
    q = np.array([[0, -1], [1, 0]])
    rotated = g.copy()
    for lo, hi in ((0, 16), (38, 166)):
        rotated[:, lo:hi] = (g[:, lo:hi].reshape(4, -1, 2)@q).reshape(4, -1)
    aa, bb = m.temporal_features(rotated, np.ones(4))
    np.testing.assert_allclose(a, aa, atol=1e-12); np.testing.assert_allclose(b, bb, atol=1e-12)


def test_history_ablation_has_no_neighbor_dependency():
    g = geometry(); a, b = m.temporal_features(g, np.ones(4))
    g[:, 38:294] = 0
    aa, bb = m.temporal_features(g, np.ones(4))
    np.testing.assert_array_equal(a, aa); assert not np.array_equal(b, bb)


@pytest.mark.parametrize('column', [23, 173])
def test_reject_future_or_asynchronous_times(column):
    g = geometry(); g[0, column] = 1
    with pytest.raises(ValueError): m.temporal_features(g, np.ones(4))


def test_fit_unknown_rows_and_inference_caps():
    rng = np.random.default_rng(4); n = 50; x = rng.normal(size=(n, 3))
    p = np.tile([1, .5, .4, .1], (n, 1)); y = p.copy(); y[:, 3] = np.maximum(x[:, 0], 0)*.2
    y[-1] = np.nan; sites = np.array(['a']*25+['b']*25); env = np.ones(n)
    model = m.fit(x, p, y, env, sites, {'outer', 'inner'})
    altered = x.copy(); altered[-1] = 1e10
    assert model == m.fit(altered, p, y, env, sites, {'outer', 'inner'})
    result = m.predict(model, x, p, env)
    np.testing.assert_array_equal(result[:, :3], p[:, :3])
    assert np.all((result[:, 3] >= 0) & (result[:, 3] <= p[:, 1]))
    assert 'target' not in inspect.signature(m.predict).parameters
    with pytest.raises(ValueError): m.fit(x, p, y, env, sites, {'a'})


def test_effective_support_does_not_count_overlap_as_independence():
    y = np.tile([2, 1, 2, 1], (4, 1)); y[-1, 3] = 10
    result = m.support(y, np.ones(4), np.array(['a']*4), np.zeros(4, int),
        np.array([1, 1, 1, 2]), np.array([0, 1, 2, 229]))[0]
    assert result['event_tracks'] == 2 and result['event_agent_queries'] == 4
    assert result['event_nonoverlap_recording_queries'] == 2
    assert result['event_track_mass_effective_count'] < 2
    assert result['intervals_are_not_independent']


def test_frozen_cap_lower_bound_is_not_prediction_feature():
    p = np.tile([2, 1, 1, .2], (3, 1)); y = p.copy(); y[:, 3] = [0, 1, 3]
    result = m.metrics(p, y, np.ones(3)*3)
    assert result['frozen_cap_unavoidable_MSE'] == pytest.approx(4/3)
    assert result['harm_above_cap_rows'] == 1
