import numpy as np

from src.evaluation.m3w_recipient_release_capacity import assess
from src.world_model.m3w_released_recipient_paths import paths
from scripts.run_m3w_recipient_release_crossfit import choose


def test_releasing_recipients_or_count_changes_feasibility():
    cv = np.zeros((3, 12, 2))
    ols = cv.copy()
    ols[:, :, 0] = 1.
    y = ols.copy()
    y[1:, :, 0] = 2.
    ols[2, :, 0] = -1.
    bank = y[:, None].copy()
    bank[2, 0, :, 0] = -5.
    valid = np.ones((3, 12), bool)
    inverse, w = np.zeros(3, int), np.ones(3)/3
    cut = {"easy": 1.5, "hard": 5.}
    args = (cv, ols, bank, y, valid, inverse)
    a = assess(*args, np.array([2]), w, "r", cut)
    assert a["status"] == "verified_both"
    assert a["witness"][0] == [0, 1]
    b = assess(*args, np.array([3]), w, "r", cut)
    assert b["status"] == "risk_infeasible"
    c = assess(*args, None, w, "r", cut)
    assert c["status"] == "verified_both"
    assert c["evidence"]["switched"] == 2


def test_nested_bank_keeps_straight_options_and_caps_curvature():
    h = np.zeros((2, 8, 2))
    h[:, :, 0] = np.arange(8)**2
    h[:, :, 1] = np.arange(8)*2
    cv = np.broadcast_to([13., 2.], (2, 12, 2))*np.arange(1., 13.)[None, :, None]
    ols = np.broadcast_to([11., 2.], (2, 12, 2))*np.arange(1., 13.)[None, :, None]
    p = paths(h, cv, ols)
    assert p.shape == (2, 10, 12, 2)
    np.testing.assert_allclose(p[:, 4], ols)
    np.testing.assert_allclose(p[:, 0], (cv+ols)/2)
    cap = np.linalg.norm(ols-cv, axis=-1)
    assert (np.linalg.norm(p[:, 8]-ols, axis=-1) <= cap+1e-12).all()


def test_predicted_joint_requires_two_supported_risk_inequalities():
    gain = np.array([[2., 1.], [1., 2.]])
    all_excess = np.full_like(gain, -1.)
    easy_excess = np.array([[1., -.5], [0., 1.]])
    ind, joint = choose(gain, all_excess, easy_excess, np.ones_like(gain, bool), np.ones(2))
    np.testing.assert_array_equal(ind, [0, 1])
    np.testing.assert_array_equal(joint, [1, 0])
    _, none = choose(gain, all_excess, np.zeros_like(gain), np.ones_like(gain, bool), np.ones(2))
    np.testing.assert_array_equal(none, [-1, -1])
