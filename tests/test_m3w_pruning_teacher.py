import numpy as np

from src.world_model.m3w_pruning_teacher import calibrate, moments, scores


def test_positive_imitation_is_not_the_same_as_gain_guard():
    x = np.eye(2)
    a = scores(x, np.array([[.7, -.1], [.5, .2]]))
    np.testing.assert_array_equal(a, [[.7, .5], [-1., .5]])
    np.testing.assert_array_equal(scores(x, None), -1.)


def test_fit_targets_and_original_gain_requirement():
    x = np.ones((2, 1))
    g, b = moments(x, np.array([0., 1.]), np.array([-4., 2.]), np.ones(2), np.ones(2))
    np.testing.assert_allclose(g, [[1.]])
    np.testing.assert_allclose(b, [[.5, -1.]])
    v = dict(all_excess=np.array([-.4, 2.]), easy_excess=np.array([-.4, 0.]),
             easy_signed=np.zeros(2), ref_easy=np.array([20., 0.]),
             remove_lower=np.array([-1., -3.]))
    rows = [([np.array([0, 1])], v, np.array([[.1, .6], [-1., -1.]]), 20., np.ones(2))]
    cut, audit = calibrate(rows, 0)
    assert cut is None
    assert audit['recover_and_preserve_risk'] > 0
    assert not audit['also_nonnegative_gain']
    v['remove_lower'][1] = 3.
    assert calibrate(rows, 0)[0] is not None
