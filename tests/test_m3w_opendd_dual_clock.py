import numpy as np

from src.world_model import m3w_opendd_dual_clock as m
from src.world_model import m3w_opendd_dual_deadline as deadline


def test_early_wait_never_forces_a_terminal_action():
    seen = []

    def score(q, p, spent, menu):
        seen.append((q, p, spent))
        return [[1., 1., -1.] if q == 0 else [1., 1., 1.] for _ in menu]

    p = [((0,), ()), ((), (1,))]
    assert m.stream(iter(p[:1]), score, "continuation") == ()
    assert m.stream(iter(p), score, "continuation") == ((1, 1),)
    assert m.stream(iter(p), score, "myopic") == ((0, 0), (1, 1))
    assert seen[-1][2] == ((0, 0),)


def test_nominations_are_distinct_and_match_role_budget():
    p = [((0,), (0,)), ((1, 2), (1,))]
    assert ((0, 0), (1, 0)) not in list(m.proposals(p[0]))
    assert ((1, 0),) not in list(m.proposals(p[0], ((0, 0),)))
    assert m.earliest(p, {1}) == ((1, 0),)
    assert m.earliest(p, {0, 1}) == ((0, 0), (1, 1))


def test_features_do_not_depend_on_unselected_or_future_rows():
    p, spent, act = ((1,), ()), ((1, 0),), ((0, 1),)
    f = [np.arange(8.).reshape(4, 2), np.arange(8.).reshape(4, 2)]
    a = m.describe(f, 1, p, spent, act)
    f[0][2:] += 1e6
    f[1][1:] += 1e6
    np.testing.assert_array_equal(m.describe(f, 1, p, spent, act), a)


def test_fixed_deadline_spends_only_current_due_roles_and_does_not_force_terminal():
    low = lambda t, p, s, a: np.full((len(a), 3), -1.)
    p = [((0,), ()), ((), (1,)), ((2,), (3,)), ((), (4,))]
    assert deadline.stream(iter(p[:2]), low) == ()
    assert deadline.stream(iter(p), low) == ((0, 2), (1, 4))
    no_gain_at_deadline = [((0,), ()), ((), ()), ((), ()), ((3,), ())]
    assert deadline.stream(iter(no_gain_at_deadline), low) == ()
