import numpy as np

from src.world_model import m3w_cv_risk_frontier as model


def test_scene_solver_can_pay_harm_with_another_agents_credit():
    gain = np.array([[[4., 5.], [0., 0.]], [[1., 1.], [0., 0.]]])
    risk = np.array([[[1., 1., 0.], [-3., -3., 0.]], [[-1., -1., 0.], [0., 0., 0.]]])
    legal = np.array([[True, False], [True, False]])
    ref = np.ones((2, 2))
    z = model.generate(gain, risk, legal, ref, 1., grid=((0., 0., .02, .05),))
    assert [p.tolist() for p in z] == [[0, 0]]


def test_risk_rounded_plan_and_unconstrained_variant_both_retained():
    # Each row can choose no edit or a positive-gain risky edit.
    x = np.array([.4, .6, .4, .6])
    offsets = np.array([0, 2, 4])
    z = np.array([[0., 0., 0., 0., 0., 0.], [3., 3., 1., 1., 0., -1.],
        [0., 0., 0., 0., 0., 0.], [5., 5., 1., 1., 0., -1.]])
    plans = model.round_support(x, offsets, z, np.array([1., 1., 0., -1.]))
    assert [p.tolist() for p in plans] == [[0, 3], [1, 3]]


def test_unavailable_candidates_cannot_be_created_by_solver():
    gain = np.ones((3, 2, 2))
    risk = np.zeros((3, 2, 3))
    legal = np.array([[False, False], [True, False], [False, True]])
    ref = np.ones((3, 2))
    out = model.generate(gain, risk, legal, ref, 1.)
    assert [p.tolist() for p in out] == [[-1, 0, 1]]
    assert model.generate(gain, risk, legal & False, ref, 1.) == []


def test_integer_audit_rejects_fractionally_feasible_slate():
    from scripts.audit_m3w_cv_risk_frontier_integer import solve
    # Each alternative harms a different risk category. Their convex
    # mixture passes but neither exact plan does.
    block = np.array([[2., 2., 1., -1., 0., 1.], [2., 2., -1., 1., 0., 1.]])
    assert solve([block], 100., 0., 0.) == (None, 'integer_infeasible_at_margin')
    block[1, 3] = -.5
    picks, status = solve([block], 100., 0., 0.)
    assert status == 'candidate'
    assert picks == [1]
