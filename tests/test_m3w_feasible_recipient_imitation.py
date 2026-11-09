import numpy as np

from src.world_model.m3w_feasible_recipient_imitation import teacher


def test_whole_record_teacher_requires_easy_support_and_distinct_recipients():
    rows, slots = np.array([0, 1, 0, 2]), np.array([0, 0, 1, 1])
    gain = np.array([[6, 6], [4, 4], [7, 7], [1, 1]], dtype=float)
    risk = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    easy = np.array([True, True, False, False])
    unconstrained, _ = teacher(rows, slots, gain, np.zeros(2), risk, easy, np.zeros(3), False)
    constrained, status = teacher(rows, slots, gain, np.zeros(2), risk, easy, np.zeros(3), True)
    assert status == 'qualified'
    assert set(unconstrained) == {1, 2}
    assert set(constrained) == {0, 3}
    assert teacher(rows, slots, gain, np.zeros(2), risk, easy & False, np.zeros(3), True) == (None, 'no_easy')
