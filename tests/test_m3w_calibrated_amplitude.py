import numpy as np

from src.evaluation.m3w_frozen_amplitude import shrink
from src.evaluation.m3w_recording_forecast_comparison import score_recording
from src.world_model import m3w_calibrated_amplitude as m


def test_matched_query_budget_retains_count_and_coefficient_sum():
    pred = np.zeros((3, len(m.ALPHA), 3))
    pred[0, 0, 0], pred[1, 2, 0] = 2., 3.
    p = m.matched(pred, [np.array([0, 1]), np.array([2])])
    np.testing.assert_array_equal(p, [0, 2, 1])
    np.testing.assert_array_equal(m.ALPHA[p], [.125, .375, .25])


def test_donor_masses_equal_original_scorer_including_unknown_radius():
    floor = np.ones((4, 2, 2))
    alt = floor.copy()
    alt[:3] *= np.array([-.5, 2., .1])[:, None, None]
    target = np.zeros_like(floor)
    valid = np.array([[True, True], [False, False], [True, False], [True, True]])
    queries = np.array([0, 0, 1, 1])
    i = np.arange(3)
    choices = np.array([0, 3, 4])
    realized = m.realized(floor, alt[i], target, valid, i, np.full(4, .5), 2.)
    selected = shrink(floor, alt, np.r_[m.ALPHA[choices], 1.])
    base = shrink(floor, alt, .25)
    cv = 'constant_velocity_causal_fd'
    out = score_recording({cv: floor, 'q': base, 'arm': selected}, {'p': ('q', 'arm', cv)},
        target, valid, queries, 'r', easy_cut=2., hard_cut=3.)
    passed, easy_bad, lower = m.observe(realized, choices)
    risk = out['metrics']['arm']['risk']
    assert passed == risk['supported']
    assert easy_bad == (risk['known_easy_net_degradation'] > .02)
    p = out['comparisons']['p']
    np.testing.assert_allclose(lower, p['known_delta']-p['unknown_radius'])
