import numpy as np

from src.world_model.m3w_vehicle_pair_angle import choose, features, WIDTH


def test_pair_rule_keeps_incumbent_when_gain_heads_disagree():
    z = np.array([[[3., -1., 4.], [-1., 3., 4.]], [[2., .3, 4.], [1., 1., 4.]]])
    single, both, score = choose(z, np.ones((2, 2, 2)))
    assert single.tolist() == [1, 1]
    assert both.tolist() == [0, 2]
    assert score[0] == 0. and score[1] > 0.


def test_pair_features_keep_nominee_order_and_compare_scales():
    x = np.arange(3*WIDTH).reshape(3, WIDTH)
    f, r = features(x, np.array([[0., 0.], [1., 2.], [3., 1.]]))
    np.testing.assert_array_equal(f[:, :WIDTH], np.stack([x[0]]*2))
    np.testing.assert_array_equal(f[:, WIDTH:2*WIDTH], x[1:])
    np.testing.assert_allclose(r, [[1/3, 1.], [1., .5]])


def test_missing_target_can_reverse_proxy_sign():
    from scripts.run_m3w_vehicle_pair_angle import scale, targets

    paths = np.zeros((4, 2, 3, 2))
    paths[1:, 0, :, 0] = np.array([2., 1., -1.])[:, None]
    valid = np.array([[True, False, False], [True, True, True]])
    label = np.full((2, 3, 2), np.nan)
    label[valid] = 0.
    y = targets(paths, label, valid, scale(paths), np.ones(4, dtype=bool))
    np.testing.assert_allclose(y[0, :4], [1., 1., -1/3, -1/3], atol=1e-7)


def test_missing_pair_targets_widen_the_bound_not_the_observed_term():
    from scripts.audit_m3w_vehicle_pair_angle_bounds import pair_bounds

    before = np.zeros((2, 2, 2))
    after = before.copy()
    after[0, :, 0] = 1.
    valid = np.array([[True, False], [True, True]])
    target = np.zeros_like(before)
    target[~valid] = np.nan
    values, n = pair_bounds(after, before, target, valid)
    np.testing.assert_allclose(values, [-1., 0., -.5, .5])
    assert n == dict(changed=2, observed_changed=1)
