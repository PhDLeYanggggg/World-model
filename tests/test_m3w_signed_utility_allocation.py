import itertools
import numpy as np
import pytest
from src.world_model import m3w_signed_utility_allocation as api


def fixture():
    p = np.zeros((2, 7, 5)); ok = np.zeros((2, 7), bool)
    # The gainful action needs a slightly harmful easy-reference donor.
    p[0, 0] = [10., 0., 1., .1, .04]
    p[1, 0] = [0., .01, 1., 2., 0.]
    ok[:, 0] = True
    return p, ok


def test_signed_utility_donor_can_enable_positive_joint_gain_without_net_harm_credit():
    p, ok = fixture(); floor = np.full(2, -1)
    choice, info = api.solve(p, ok, floor, exact_count=None, seconds=2.)
    assert choice.tolist() == [0, 0] and info['optimal']
    assert info['selected_nonpositive_utility_actions'] == 1
    assert api.base.terms(p, choice)[0] == 9.99
    assert max(api.base.terms(p, choice)[1]) <= 0
    assert not api.base.feasible(p, ok, choice)


def test_matched_zero_count_cannot_invent_a_support_anchor():
    p, ok = fixture(); choice, info = api.solve(p, ok, np.full(2, -1), exact_count=0, seconds=2.)
    assert choice.tolist() == [-1, -1] and info['optimal']


def test_budget_uses_positive_harm_not_net_benefit_as_credit():
    p, ok = fixture(); p[0, 0, 1] = 1.
    c = np.array([0, 0]); assert not api.feasible(p, ok, c)


def test_unconstrained_solution_does_not_include_negative_utility_rows():
    p, ok = fixture(); p[0, 0, 4] = 0
    c, info = api.solve(p, ok, np.full(2, -1), exact_count=None, seconds=2.)
    assert c.tolist() == [0, -1] and info['optimal']


def test_solver_failure_retains_feasible_anchor(monkeypatch):
    p, ok = fixture()
    class Failed:
        status = 1
        x = None
    monkeypatch.setattr(api, 'milp', lambda **_: Failed())
    c, info = api.solve(p, ok, np.full(2, -1), exact_count=None, seconds=2.)
    assert (c == -1).all() and not info['optimal'] and not info['accepted']


def test_wrong_count_or_illegal_incumbent_refuses():
    p, ok = fixture()
    with pytest.raises(ValueError): api.solve(p, ok, np.array([1, -1]), exact_count=1, seconds=2.)
    with pytest.raises(ValueError): api.solve(p, ok, np.full(2, -1), exact_count=1, seconds=2.)


def test_small_random_exact_optima_and_nesting():
    rng = np.random.default_rng(314)
    for _ in range(60):
        p = np.zeros((3, 7, 5)); p[:, :2, :2] = rng.uniform(0, 2, (3, 2, 2))
        p[:, :2, 2:4] = rng.uniform(.1, 60, (3, 2, 2))
        p[:, :2, 4] = p[:, :2, 1]*rng.uniform(0, 1, (3, 2))
        ok = np.zeros((3, 7), bool); ok[:, :2] = True
        floor = np.full(3, -1); c, info = api.solve(p, ok, floor, exact_count=None, seconds=2.)
        scores = [api.base.terms(p, np.array(x))[0] for x in itertools.product((-1, 0, 1), repeat=3)
            if api.feasible(p, ok, np.array(x))]
        assert api.feasible(p, ok, c) and info['optimal']
        assert np.isclose(api.base.terms(p, c)[0], max(scores), atol=1e-10)


def test_allocation_resume_is_exact_and_causal_bindings_refuse_changes():
    p, ok = fixture(); args = (p, ok, np.array(['r', 'r']), np.array([1, 1]), np.array([0, 1]), np.full(2, -1))
    saved = []
    direct = api.allocate(*args, seconds=2., checkpoint=saved.append)
    resumed = api.allocate(*args, seconds=2., resume_state=saved[-1])
    for k in direct[0]: np.testing.assert_array_equal(direct[0][k], resumed[0][k])
    assert direct[1] == resumed[1]
    changed = p.copy(); changed[0, 0, 0] += 1
    with pytest.raises(ValueError): api.allocate(changed, *args[1:], seconds=2., resume_state=saved[-1])
