import numpy as np

from src.world_model.m3w_easy_slot_support import guard, prefix_features, targets


def test_guard_reverts_whole_budget_block_when_support_falls():
    cv = np.zeros((4, 2, 2))
    fixed, pooled = cv.copy(), cv.copy()
    fixed[[0, 2], :, 0] = .5
    pooled[[1, 3], :, 0] = .5
    output, stats = guard(cv, fixed, pooled, np.array([.4, .2, .2, .4]), [[0, 1], [2, 3]])
    np.testing.assert_array_equal(output[:2], fixed[:2])
    np.testing.assert_array_equal(output[2:], pooled[2:])
    assert stats == {'changed_blocks': 2, 'rejected_blocks': 1}


def test_membership_matches_original_positive_known_easy_definition():
    cv = np.zeros((4, 2, 2))
    y = cv.copy()
    y[:, :, 0] = np.asarray([.2, .6, 0., .1])[:, None]
    valid = np.ones((4, 2), bool)
    valid[3] = False
    e, mass = targets(cv, y, valid, .2)
    np.testing.assert_array_equal(e, [True, False, False, False])
    np.testing.assert_allclose(mass, [.2, 0., 0., 0.])


def test_prefix_features_are_translation_invariant():
    h = np.arange(4*8*2, dtype=float).reshape(4, 8, 2)
    q = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(prefix_features(h, q), prefix_features(h+np.array([20., -10.]), q))
