import numpy as np

from src.world_model import m3w_opendd_two_reserves as m


def pack(rows, gain, easy):
    return tuple(map(np.array, (rows, gain, easy)))


def test_independent_slots_do_not_anticipate_later_gain():
    first = pack([1, 2], [-.1, -.2], [.1, .8])
    late = pack([3], [.4], [.9])
    assert m.choose(iter([first])) == (None, 2, 1)
    assert m.choose(iter([first, late])) == (3, 2, 1)


def test_second_slot_matches_time_without_overwriting_gain():
    first = pack([1, 2, 3], [.7, .4, .1], [.95, .1, .8])
    assert m.choose(iter([first])) == (1, 3, 2)
    assert m.choose(iter([pack([1], [.7], [.95])])) == (1, None, None)


def test_constrained_ranking_keeps_easy_requirement_and_distinct_recipient():
    from scripts.audit_m3w_opendd_two_reserve_ranking import rerank

    assert rerank(*pack([1, 2, 3, 4], [.99, .9, .1, .4], [.9, .49, .8, .7]), 1) == 4
