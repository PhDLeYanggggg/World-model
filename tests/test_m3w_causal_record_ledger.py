import itertools

import numpy as np

from src.world_model import m3w_causal_record_ledger as m


def test_query_solver_matches_exhaustive_small_cases():
    rng = np.random.default_rng(21)
    for _ in range(16):
        value, a, e = rng.normal(size=(4, 3, 3)).transpose(2, 0, 1)
        allowed = rng.random((4, 3)) > .15
        slack = rng.random(2)
        for count in (None, 0, 1, 2):
            feasible = []
            for c in itertools.product(range(-1, 3), repeat=4):
                c = np.array(c); ix = np.flatnonzero(c >= 0)
                if len(ix) > 2 or count is not None and len(ix) != count or not allowed[ix, c[ix]].all():
                    continue
                t = m.totals(c, value, a, e)
                if np.all(t[1:] <= slack):
                    feasible.append(t[0])
            answer = m.query_choice(value, a, e, allowed, slack, count)
            if feasible:
                np.testing.assert_allclose(m.totals(answer, value, a, e)[0], max(feasible))
            else:
                assert answer is None


def test_past_balance_can_enable_but_future_balance_cannot():
    groups = [np.arange(2), np.arange(2, 4), np.arange(4, 6)]
    v = np.array([[1., 0.], [0., 0.], [9., 1.], [0., 0.], [99., 1.], [0., 0.]])
    a = np.array([[-2., 0.], [0., 0.], [1., 0.], [0., 0.], [-10., 0.], [0., 0.]])
    ok = np.ones_like(v, bool)
    choices, meta = m.stream(groups, v, a, a, ok, counts=[1, 1, 1])
    local, _ = m.stream(groups, v, a, a, ok, counts=[1, 1, 1], carry=False)
    prefix, _ = m.stream(groups[:2], v[:4], a[:4], a[:4], ok[:4], counts=[1, 1])
    assert choices[2] == 0 and local[2] == 1 and meta['queries_using_credit'] == 1
    np.testing.assert_array_equal(prefix, choices[:4])


def test_negative_utility_action_can_fund_current_pair():
    v = np.array([[4.], [-1.], [.2]])
    a = np.array([[2.], [-2.], [0.]])
    c = m.query_choice(v, a, a, np.ones_like(v, bool), np.zeros(2))
    np.testing.assert_array_equal(c, [0, 0, -1])
