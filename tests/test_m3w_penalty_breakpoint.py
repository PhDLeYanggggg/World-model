import pytest

from src.evaluation.m3w_penalty_breakpoint import breakpoint


def test_breakpoint_reproduces_winner_before_and_after_exact_crossing():
    u, p = [5., 4., 2.], [3., 1., 0.]
    early = breakpoint(u, p, .1)
    assert early == dict(base=0, winner=0, threshold=.5, penalty_at_base=3., alternatives_with_less_penalty=2)
    assert breakpoint(u, p, .5)['winner'] == 0
    assert breakpoint(u, p, .51)['winner'] == 1


def test_minimum_penalty_winner_and_additive_tie():
    assert breakpoint([5., 4.], [0., 0.], .1)['threshold'] is None
    tied = breakpoint([5., 5.], [1., 0.], .1)
    assert tied['threshold'] == 0 and tied['winner'] == 1


def test_negative_penalty_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        breakpoint([1., 0.], [0., -1.], .1)
