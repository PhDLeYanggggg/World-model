import numpy as np

from src.evaluation.m3w_opendd_location import paired
from src.world_model import m3w_tumdot_matched_selector as m


def test_fit_roster_excludes_both_intersection_and_day():
    keys = [f"P{s}-d{d}" for s in range(1, 4) for d in range(1, 4)]
    assert m.fit_roster("P2-d1", keys) == ("P1-d2", "P1-d3", "P3-d2", "P3-d3")


def test_partial_and_wholly_unknown_targets_reproduce_stepwise_lower():
    cv = np.zeros((3, 12, 2))
    b = np.stack([cv + [1., 0.], cv - [1., 0.]], axis=1)
    valid = np.array([[True]*12, [True]*6+[False]*6, [False]*12])
    y = np.where(valid[..., None], np.broadcast_to([1., 0.], cv.shape), np.nan)
    z = m.targets(cv, b, y, valid, np.ones(3), 1.5)
    for i in range(2):
        np.testing.assert_allclose(z[:, i, 0], paired(cv, b[:, i], y, valid)[1].mean(1))
    np.testing.assert_array_equal(z[:, 1, 2], [1., 1., 0.])


def test_past_probe_is_translation_invariant_and_uses_matching_candidates():
    h = np.zeros((2, 8, 2))
    h[:, :, 0] = np.arange(8)
    cv = np.zeros((2, 12, 2))
    cv[:, :, 0] = np.arange(1, 13)
    bank = np.stack([cv+[.2, 0.], cv-[.2, 0.]], axis=1)
    x, probe = m.features(h, cv, cv, bank, np.full(2, .2))
    z, p = m.features(h+1000, cv, cv, bank, np.full(2, .2))
    np.testing.assert_array_equal(x, z)
    np.testing.assert_array_equal(probe, p)
    np.testing.assert_allclose(probe, -1.)
