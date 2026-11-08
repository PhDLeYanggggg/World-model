import numpy as np

from src.world_model import m3w_opendd_release_credit as model
from scripts.run_m3w_opendd_release_credit import solve


def packet(rows, release, value, support):
    return np.array(rows), release, np.array(value), np.array(support)


def test_support_wait_uses_no_future_and_remembers_original_credit_owner():
    rows = [packet([0, 1], 1, [2., 1.], [.1, .2]),
            packet([2, 3], 0, [3., 2.], [.7, .9]),
            packet([4], 0, [1.], [.5])]
    e, unused = model.stream(iter(rows), "support_wait")
    assert (e, unused) == ([(2, 0, 1, 1)], 0)
    prefix, _ = model.stream(iter(rows[:1]), "support_wait")
    assert prefix == [a for a in e if a[2] < 1]
    assert model.stream(iter(rows), "immediate")[0] == [(0, 0, 0, 0)]


def test_no_borrowing_expiry_and_local_spare_cap():
    rows = [packet([0, 1], 2, [-1., -2.], [.8, .9]),
            packet([2, 3], 0, [-1., -2.], [.8, .9]),
            packet([4, 5], 0, [-1., -2.], [.8, .9])]
    e, expired = model.stream(iter(rows), "value_wait")
    assert (e, expired) == ([(4, 0, 2, 2)], 1)
    assert not model.legal_schedule([1, 0, 0], [0, 0, 1], [1, 1, 1], 1, True)
    assert model.legal_schedule([1, 0, 0], [0, 0, 1], [1, 1, 1], 2, True)
    assert not model.legal_schedule([0, 1], [1, 0], [1, 1], 2, True)


def test_exact_delay_recovers_support_without_inventing_credit():
    groups = np.arange(3)
    count = np.array([1, 0, 0])
    gain = np.ones((3, 2))
    risk = np.zeros((3, 3))
    easy = np.array([False, False, True])
    cap = np.ones(3, dtype=int)
    assert solve(groups, count, cap, gain, risk, easy, 0., 1., 1)[0] == "no_eligible_easy"
    status, take = solve(groups, count, cap, gain, risk, easy, 0., 1., 2)
    assert status == "qualified"
    np.testing.assert_array_equal(take, [False, False, True])
