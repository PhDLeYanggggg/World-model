import numpy as np

from src.world_model import m3w_directional_utility as model


def test_decomposition_with_missing_steps_and_zero_reference():
    c = np.zeros((1, 3, 2))
    y = np.array([[[2., 1.], [0., 0.], [np.nan, np.nan]]])
    valid = np.array([[True, True, False]])
    b = np.array([[[[.1, 1.], [.4, 1.], [70., 30.]]]])
    g, q = model.targets(c, y, valid, np.eye(2)[None], b, np.ones(1))
    linear = (g[:, None, :2] * b[:, :, :2]).sum(-1).mean(-1)
    direct = (np.linalg.norm(y[:, :2], axis=-1) -
              np.linalg.norm(b[:, :, :2] - y[:, None, :2], axis=-1)).mean(-1)
    np.testing.assert_allclose(linear - q, direct)
    assert (q >= 0).all()


def test_nonstationary_features_rotate_together():
    h = np.array([[[-3., -1.], [-1., -1.], [0., 0.]]])
    c = np.ones((1, 4, 2))
    b = (c + np.array([.2, .4]))[:, None]
    q = np.array([[0., -1.], [1., 0.]])
    args = (np.ones(1), np.ones(1))
    a = model.features(h, *args, b, c)
    d = model.features(h @ q, *args, b @ q, c @ q)
    np.testing.assert_allclose(a[0], d[0], atol=1e-12)
    np.testing.assert_allclose(a[1], d[1])


def test_coherent_gain_matches_original_decoder():
    from src.world_model import m3w_record_joint_actions as original

    rng = np.random.default_rng(18)
    paths = rng.normal(size=(9, 3, 12, 2))
    floor = rng.normal(size=(9, 12, 2))
    x = rng.normal(size=(9, 3, 6))
    weights = rng.normal(size=(3, 6, 6))
    scale = rng.uniform(1., 5., size=9)
    g = np.einsum('nad,ad->na', x, weights[..., 0]) * scale[:, None]
    got = model.coherent_value(g, x, weights, scale, paths - floor[:, None])
    expected = original.decision_costs(x, weights, paths, floor, scale)[0] / scale[:, None]
    np.testing.assert_allclose(got, expected)
