import numpy as np

from src.world_model import m3w_opendd_hardness_gate as model


def test_matched_control_preserves_every_query_count_and_legality():
    q = np.array([0, 0, 0, 1, 1, 2])
    valid = np.array([True, True, False, True, True, True])
    selected = np.array([True, False, False, False, True, False])
    rank = np.array([1., 5., 99., 4., 1., 99.])
    np.testing.assert_array_equal(model.matched_count(q, selected, valid, rank),
                                  [False, True, False, True, False, False])


def test_gate_and_features_are_causal_and_invariant_to_planar_pose():
    h = np.random.default_rng(23).normal(size=(18, 8, 2)).cumsum(1)
    r = np.array([[0., 1.], [-1., 0.]])
    np.testing.assert_allclose(model.features(h), model.features(h@r+50.), atol=1e-6)
    p = np.array([[.24, .26, .5], [.25, .15, .6], [.1, .3, .6], [.05, .46, .49]])
    np.testing.assert_array_equal(model.decide(p, np.array([1, 1, 0, 1], bool)), [True, False, False, False])
