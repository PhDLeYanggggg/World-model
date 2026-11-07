import numpy as np

from src.evaluation import m3w_slate_geometry_capacity as m


def test_augmentation_does_not_add_recipients_or_count_noops_as_interventions():
    cv = np.zeros((4, 12, 2))
    ols = cv.copy()
    ols[[0, 2, 3], :, 0] = 3.
    kinematic = cv.copy()
    slots = np.full((4, 2, 2), -1)
    slots[0] = [[0, 1], [2, 3]]
    paths = np.zeros((*slots.shape, 12, 2))
    paths[0, :, :, :, 0] = 1.
    ids, alternatives = m.augment(cv, ols, kinematic, [0, 2, 4], slots, paths)
    np.testing.assert_array_equal(ids[:4], slots)
    assert (ids[4, 0] == -1).all()
    np.testing.assert_array_equal(ids[4, 1], slots[0, 1])
    np.testing.assert_array_equal(ids[8], slots[0])
    np.testing.assert_allclose(alternatives[8, 0, :, :, 0],
                               np.broadcast_to(np.array([2., -1.])[:, None], (2, 12)))
    np.testing.assert_allclose(alternatives[8, 1, :, :, 0], 1.)


def test_adding_ols_trajectory_can_restore_risk_support_with_same_recipient():
    cv = np.zeros((2, 12, 2))
    y = cv.copy()
    y[:, :, 0] = 1.
    ols = y.copy()
    slots = np.full((4, 1, 2), -1)
    slots[3, 0, 0] = 0
    paths = np.zeros((*slots.shape, 12, 2))
    paths[3, 0, 0, :, 0] = -1.
    out = m.audit(cv, ols, cv, y, np.ones((2, 12), bool), np.array([0, 2]),
                  slots, paths, 'sample', 1.5, 3.)['outcomes']
    assert out['frozen']['proved_infeasible']
    assert out['plus_ols4']['verified_witness']
    assert out['plus_ols4']['added_queries_in_witness'] == 1
    assert not out['plus_ols4']['witness_positive']
