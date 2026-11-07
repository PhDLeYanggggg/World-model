import numpy as np

from src.world_model.m3w_recording_witness_retrieval import teacher


def test_label_requires_positive_recording_gain_and_no_unobserved_recipient():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.arange(1., 5.)[:, None]
    valid = np.ones((4, 12), bool)
    valid[-1] = False
    bank = y[:, None]
    o, cap, cut = np.array([0, 4]), np.array([2]), dict(easy=1.5, hard=2.)
    take, out = teacher(cv, cv, bank, y, valid, o, cap, 'x', cut)
    np.testing.assert_array_equal(take, [True, False, True, False])
    assert out['positive'] and out['selected'] == 2
    take, out = teacher(cv, y, bank, y, valid, o, cap, 'x', cut)
    assert not out['positive'] and not take.any() and out['verified_feasible']
