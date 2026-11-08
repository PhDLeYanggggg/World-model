import numpy as np

from src.world_model import m3w_donor_shape_choice as model
from scripts.run_m3w_donor_shape_ceiling import best


def test_risk_penalty_and_threshold_preserve_anchor():
    z = np.array([[.2, .3, .3, -.9], [.2, -.1, 0., -.9], [0., 0., 0., 0.]])
    np.testing.assert_array_equal(model.use_donor(z, None), [False, False, False])
    np.testing.assert_array_equal(model.use_donor(z, (.01, 1.)), [False, True, False])
    np.testing.assert_array_equal(model.use_donor(z, (.01, 0.)), [True, True, False])


def test_validation_rejects_gain_with_harm_without_redefining_risk():
    pred = np.array([[.6, 0., 0., 0.]])
    y = np.array([[2., -1., .1, -.1]])
    cells = {"A": [(pred, y, 10.)], "B": [(pred, y, 12.)]}
    assert model.calibrate(cells, guarded=False)[0] is not None
    assert model.calibrate(cells, guarded=True)[0] is None


def test_calibration_macro_averages_sites_not_windows():
    pred = np.array([[.6, 0., 0., 0.]])
    cells = {"small": [(pred, np.array([[1., 0., 0., 0.]]), 1.)],
             "large": [(np.tile(pred, (10, 1)), np.tile([-2., 0., 0., 0.], (10, 1)), 100.)]}
    rule, table = model.calibrate(cells, guarded=True)
    assert rule is not None
    np.testing.assert_allclose(table[1]["gain"], 40.)


def test_oracle_separates_utility_from_risk_and_verifies_large_case():
    for n in (2, 13):
        gain = np.ones((n, 2))
        risk = np.ones((n, 3))
        base = np.array([-.5, -.5, 0., 0., 0.])
        assert best(gain, risk, base, np.zeros(3))[0] == "no_positive"
        assert best(gain, risk, base, np.zeros(3), safety=False)[0] == "qualified"
        status, take = best(gain, risk, base, np.ones(3))
        assert status == "qualified"
        assert take.sum() == 1
