import numpy as np

from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_donor_risk_readout as m


def test_donor_weights_and_constant_no_event_support():
    def site(n, event):
        y = np.zeros((n, 10))
        y[:, 2] = event
        return np.zeros((n, 4)), np.zeros((n, 2)), y

    fit = m.fit([site(10, 1.), site(100, -1.)])
    np.testing.assert_allclose(fit["risk"][0]["prior"], .5)
    assert fit["risk"][1]["sign"] == {"constant": 0.}
    z = m.apply(fit, np.zeros((2, 3, 4)), np.zeros((2, 3, 2)))
    np.testing.assert_allclose(z["prob"][..., 0], .5, atol=1e-6)
    assert np.isfinite(z["size"]).all()


def test_calibrated_policy_preserves_fallback_and_uses_harm_as_penalty():
    g = np.array([[[2., 1.], [2., 1.], [2., 1.]]])
    risk = np.zeros((1, 3, 3))
    risk[0, 2, 1] = .5
    s = m.score(g, risk, np.ones((1, 3)))
    assert s[0, 0] == 0. and s[0, 1] > 0. and s[0, 2] < 0.


def test_recording_acceptance_is_nonmonotone_even_for_nested_selected_sets():
    floor = np.zeros((3, 1, 2))
    y = np.repeat([[[1., 0.]]], 3, axis=0)
    candidate = np.array([[[-.03, 0.]], [[2., 0.]], [[-.06, 0.]]])
    obs = []
    for n in (1, 2, 3):
        pred = floor.copy()
        pred[:n] = candidate[:n]
        obs.append(policy(pred, floor, floor, y, np.ones((3, 1), dtype=bool), np.array(["r"]*3),
                          easy_cut=2., hard_cut=3.)["recordings"]["r"])
    np.testing.assert_allclose([r["completion_easy_risk_upper"] for r in obs], [.03, .015, .03])
    assert [r["supported"] for r in obs] == [False, True, False]
