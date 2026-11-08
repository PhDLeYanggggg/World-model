import numpy as np

from src.world_model import m3w_opendd_support_schedule as m


def packet(ids, gain, easy, prob=None):
    return tuple(map(np.asarray, (ids, gain, easy, easy if prob is None else prob)))


def test_stream_spends_support_first_and_never_forces_end_release():
    early = packet([0], [1.], [.1])
    support = packet([1], [3.], [.9])
    late = packet([2], [2.], [.1])
    assert m.stream([early]) == (None, None)
    assert m.stream([early, support]) == (None, 1)
    assert m.stream([early, support, late]) == (2, 1)
    assert m.earliest_gain([early, support, late], excluded=1) == 0
    assert m.stream([packet([2, 3], [3., 1.], [.9, .1])]) == (3, 2)


def test_learned_support_gate_sees_current_eligible_recipient_only():
    p = packet([0, 1, 2], [1., 3., 0.], [.9, .3, .6], [.1, .99, .7])
    assert m.stream([p]) == (1, 0)
    assert m.stream([p], learned=True) == (1, 2)


def test_outcome_aware_witness_preserves_budget_and_separates_order():
    big = (np.array([[1., 1.], [-2., -2.]]), np.zeros((2, 3)), np.zeros(2), np.ones(2))
    small = (np.array([[-2., -2.], [1., 1.]]), np.zeros((2, 3)), np.array([0., 1.]), np.ones(2))
    q = np.array([0, 1])
    legal = np.ones(2, dtype=bool)
    assert m.witness(big, small, q, legal, (1, 1), False, 1., 0.) == (0, 1)
    assert m.witness(big, small, q, legal, (1, 1), True, 1., 0.) is None
    assert m.witness(big, small, q, legal, (0, 1), True, 1., 0.) == (None, 1)
    assert m.witness(big, small, q, legal, (0, 1), True, 0., 0.) is None
    assert m.witness(big, small, q, legal, (0, 0), True, 1., 0.) is None
