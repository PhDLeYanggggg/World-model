import numpy as np
from scripts.audit_m3w_moving_visual_risk import score


def test_unknown_conditional_targets_do_not_become_zero_error_labels():
    truth = np.zeros((3, 5)); truth[2] = 123456
    p = np.ones((3, 5)); known = np.array([True, True, False])
    a = score(p, truth, np.array([1., .5, 0.]), known, np.ones(3))
    truth[2] = 0
    b = score(p, truth, np.array([1., .5, 0.]), known, np.ones(3))
    assert a == b
    assert all(v == 1 for v in a['conditional_MSE'].values())
    assert np.isclose(a['observation_Brier'], .25/3)
