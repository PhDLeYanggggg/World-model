import numpy as np

from src.world_model import m3w_token_count_teacher as m


def test_token_teacher_finds_later_action_not_early_high_gain():
    legal = np.tile([True, True, False], (5, 1))
    gain = np.zeros((5, 3, 2))
    gain[0, 1] = 10.
    gain[4, 1] = 1.
    risk = np.zeros((5, 3, 3))
    easy = np.zeros((5, 3), dtype=bool)
    easy[:, 1] = True
    count, status = m.teacher(gain, risk, easy, legal, 0., 1., 100.)
    assert status == "verified_witness"
    assert count.tolist() == [0, 0, 0, 0, 1]


def test_training_balance_is_predecision_and_projection_cannot_borrow():
    legal = np.tile([True, True, False], (6, 1))
    requested = np.array([1, 0, 0, 0, 1, 1])
    realized, state = m.teacher_features(requested, legal, np.array(["a"]*6))
    assert realized.tolist() == [0, 0, 0, 0, 1, 0]
    assert state.tolist() == [1, 2, 3, 4, 5, 1]
