import numpy as np

from src.world_model.m3w_nested_recipient_calibration import events, pick, roles, select
from scripts.audit_m3w_nested_recipient_frontier import exact_frontier
from scripts.audit_m3w_nested_calibration_support import choose_known


def test_roles_disjoint_and_cover_original_group_set():
    groups = [str(i) for i in range(9)]
    for held in groups:
        fit, cal = roles(groups, held)
        assert (len(fit), len(cal)) == (6, 2)
        assert set(fit).isdisjoint(cal)
        assert held not in (*fit, *cal)


def test_calibration_requires_actual_support_both_references_and_actions():
    grid = (0., .2, .5, .7)
    rows = [
        dict(threshold=0., views=3, passes=2, switches=4, gain_both=[3., 5.]),
        dict(threshold=.2, views=3, passes=3, switches=5, gain_both=[2., -1.]),
        dict(threshold=.5, views=3, passes=3, switches=4, gain_both=[1., 2.]),
        dict(threshold=.7, views=3, passes=3, switches=0, gain_both=[5., 5.]),
    ]
    assert select(rows, grid) == .5
    assert select(rows[:2], grid) is None


def test_event_targets_and_strict_query_selection():
    y = events(np.array([1., -.1, 0., 0.]), np.array([3., 1., 0., 0.]),
               np.array([True, True, False, True]), np.array([True, True, False, False]))
    np.testing.assert_array_equal(y, [[1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 0]])
    inv, quota = np.array([0, 1, 0, 1]), np.ones(2, int)
    x = np.array([.8, .1, .5, .2])
    got = pick(x, np.ones(4, bool), inv, quota, .4)
    np.testing.assert_array_equal(got, [True, False, False, False])
    x[inv == 1] = 100.
    np.testing.assert_array_equal(pick(x, np.ones(4, bool), inv, quota, .4)[inv == 0], got[inv == 0])


def test_exact_frontier_does_not_split_a_score_tie():
    b = dict(inverse=np.arange(3), counts=np.ones(3, int), weight=np.ones(3), source="s")
    m = dict(ref_easy=np.ones(3), all_excess=np.array([-.1, -.1, .3]), easy_excess=np.array([-.1, -.1, .3]),
             easy_signed=np.zeros(3), utility=np.ones(3))
    s = np.array([.8, .8, .1])
    c = (np.ones(3, bool), (m, 0., np.ones(3), 3.))
    assert exact_frontier([b], [s], [c]) == dict(levels=2, full_risk=1, qualifies=1, maximum_passes=1)


def test_hindsight_support_witness_adds_one_anchor_without_changing_quota():
    gain = np.array([1., -.01, 2., .4])
    anchor = np.array([False, True, False, False])
    q = np.array([0, 0, 1, 1])
    selected = choose_known(gain, anchor, q, np.ones(2, int), np.ones(4))
    np.testing.assert_array_equal(selected, [False, True, True, False])
    assert choose_known(gain, ~np.ones(4, bool), q, np.ones(2, int), np.ones(4)) is None
