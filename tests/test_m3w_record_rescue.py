import numpy as np

from src.world_model import m3w_record_rescue as m


def test_rescue_label_distinguishes_local_value_from_recording_repair():
    direct = np.array([0, -1, -1])
    atoms = np.array([[[10., 0., .25, 0., 0., -.25]],
                      [[1., 1., 0., 0., -.1, .1]],
                      [[4., 4., 0., 0., -.1, .1]]])
    target, mass, full = m.supervision(atoms, direct)
    assert not m.supported(mass, full)
    np.testing.assert_array_equal(target[1:, 0], [[True, False], [True, True]])
    atoms[2, 0, -1] = -.01
    assert not m.supervision(atoms, direct)[0][2, 0, 1]


def test_prefix_does_not_change_at_or_before_modified_query():
    groups = [np.array([0, 1]), np.array([2]), np.array([3])]
    direct = np.array([0, -1, 0, -1])
    predicted = np.ones((4, 1, 3))
    before = m.prefix(groups, direct, predicted)
    predicted[2] = 100.
    after = m.prefix(groups, direct, predicted)
    np.testing.assert_array_equal(before[:3], after[:3])
    assert not np.array_equal(before[3], after[3])
    np.testing.assert_array_equal(before[0], np.zeros(5))


def test_arms_share_roster_and_local_guard():
    groups = [np.array([0, 1])]
    legal = np.ones((2, 1), bool)
    scores = np.array([[[.8, .1, .7]], [[.5, .9, .1]]])
    assert next(m.options(groups, legal, scores, 0)) == (0, 0, .8, .8)
    assert next(m.options(groups, legal, scores, 1)) == (1, 0, .5, .9)
    legal[1, 0] = False
    assert next(m.options(groups, legal, scores, 1)) == (0, 0, .8, .1)
