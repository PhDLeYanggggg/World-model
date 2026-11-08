import numpy as np

from src.world_model import m3w_opendd_candidate_advantage as model
from scripts.run_m3w_opendd_advantage_capacity import solve
from scripts.run_m3w_opendd_candidate_advantage import targets


def test_fit_roster_excludes_every_producer_target_and_outer_site():
    pairs = dict(model.ordered_pairs(("c", "a", "b", "d")))
    assert len(pairs) == 6
    for excluded, fit in pairs.items():
        assert not set(excluded).intersection(fit)
        assert len(fit) == 2
    assert pairs[("a", "c")] == ("b", "d")


def test_penalty_charges_positive_risk_without_using_support_as_gain():
    y = np.array([[2., .1, .2, -.1, 1.], [1., -.2, -.3, -.5, -1.]])
    a, b = model.scores(y, np.array([2., 1.]))
    np.testing.assert_allclose(a, [4., 1.])
    np.testing.assert_allclose(b, [2., 1.])


def test_features_rotate_and_ignore_world_origin():
    rng = np.random.default_rng(82)
    h = rng.normal(size=(10, 8, 2)).cumsum(1)
    cv, alt = rng.normal(size=(2, 10, 12, 2))
    r = np.array([[0., -1.], [1., 0.]])
    a, scale = model.features(h, cv, alt)
    b, s = model.features(h@r+30., cv@r, alt@r)
    np.testing.assert_allclose(a, b, atol=2e-6)
    np.testing.assert_allclose(scale, s, atol=1e-12)


def test_exact_count_capacity_can_choose_support_without_changing_query_budget():
    groups, counts = np.array([0, 0, 1, 1]), np.array([1, 1])
    gain = np.array([[3., 3.], [2., 2.], [1., 1.], [0., 0.]])
    risk = np.zeros((4, 3))
    risk[0, 0] = 1.
    support = np.array([False, True, False, False])
    status, take = solve(groups, counts, gain, risk, support, 0., 1.)
    assert status == "qualified"
    np.testing.assert_array_equal(take, [False, True, True, False])
    assert solve(groups, counts, gain, risk, np.zeros_like(support), 0., 1.)[0] == "no_eligible_easy"
    risk[1, 0] = 1.
    assert solve(groups, counts, gain, risk, support, 0., 1.)[0] == "risk_infeasible"


def test_target_keeps_unobserved_debits_distinct_from_easy_support():
    cv = np.zeros((2, 12, 2))
    p = cv + [1., 0.]
    y = np.full_like(p, np.nan)
    y[1, -1] = p[1, -1]
    valid = np.isfinite(y).all(2)
    out = targets(dict(y=y, valid=valid), cv, p, 1., np.ones(2))
    np.testing.assert_allclose(out[0], [-1., 1., 1., 0., 0.])
    np.testing.assert_allclose(out[1], [-10./12., -.02, -.02, -1., 1.], atol=1e-7)
