import numpy as np

from src.world_model import m3w_spline_slot_choice as model


def test_joint_uses_only_same_query_tradeoff():
    p = np.array([[.4, 1., 1., 1.], [.25, -1., -1., -1.]])
    s = np.ones(2)
    q = np.array([1, 1])
    np.testing.assert_array_equal(model.choose(p, q, s, (0., 1.), False), [False, True])
    np.testing.assert_array_equal(model.choose(p, q, s, (0., 1.), True), [True, True])
    np.testing.assert_array_equal(model.choose(p, [1, 2], s, (0., 1.), True), [False, True])


def test_guarded_calibration_does_not_choose_real_risk_increase():
    data = {'a': [(np.array([[.5, 0., 0., 0.]]), np.array([0]), np.ones(1),
                   np.array([[1., 1., 0., 0.]]), 1.)]}
    assert model.calibrate(data, 'gain')[0] is not None
    assert model.calibrate(data, 'independent')[0] is None
    assert model.calibrate(data, 'joint')[0] is None
