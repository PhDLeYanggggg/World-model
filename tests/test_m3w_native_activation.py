import numpy as np

from src.world_model import m3w_native_activation as m


def test_calibration_and_evaluation_sources_are_not_in_fitting():
    train, cal = m.split(list('abcdefghijklmn'), 'm')
    assert len(train) == 10 and cal == ['n', 'a', 'b']
    assert not (set(train) & set(cal)) and 'm' not in train


def test_causal_mask_is_prefix_invariant_and_never_accepts_ineligible():
    z = np.array([.8, .3, -np.inf, .6])
    cut = m.threshold(z, 2)
    np.testing.assert_array_equal(m.causal_mask(z, cut), [True, False, False, True])
    np.testing.assert_array_equal(m.causal_mask(z[:2], cut), m.causal_mask(z, cut)[:2])
    assert not m.causal_mask(z, m.threshold(z, 0)).any()
    tied = np.array([.8, .7, .7, .5])
    np.testing.assert_array_equal(m.causal_mask(tied, m.threshold(tied, 2)), [True, False, False, False])


def test_offline_matching_separates_recordings_and_agent_counts():
    z = np.array([.1, .4, .8, .9, .2])
    old = np.array([True, False, True, False, False])
    count = np.array([2, 2, 1, 2, 2])
    rec = np.array(list('aaabb'))
    np.testing.assert_array_equal(m.matched_mask(z, old, count, rec), [False, True, True, False, False])
