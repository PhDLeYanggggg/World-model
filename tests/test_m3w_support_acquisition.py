import numpy as np

from src.evaluation.m3w_support_acquisition import pareto, solve


def test_pareto_only_within_query_and_keeps_gain_cost_tradeoff():
    q = np.array([0, 0, 1, 0, 1])
    e = np.array([-.5, -.2, -.6, -.1, -.5])
    gain = np.array([1., 0., 1., 2., 5.])
    np.testing.assert_array_equal(pareto(q, e, gain, np.ones(5, bool)), [0, 3, 2, 4])


def test_future_anchor_cannot_fund_earlier_deletion():
    p = dict(selected_query=np.array([0]), selected_cost=np.array([[3.], [0.], [0.]]),
             deletion_gain=np.array([1.]), query=np.array([1]), cost=np.array([[-.5], [-.5], [-1.]]),
             gain=np.array([1.]), shortlist=np.ones(1, bool), cap=np.array([2, 2]), full_easy=25.)
    assert solve(p, 'insert_only')[1] == 'infeasible'
    assert solve(p, 'ordered_joint')[1] == 'infeasible'
    assert solve(p, 'early_joint')[1] == 'no_eligible_anchor'
    (drop, add), status, _ = solve(p, 'free_joint')
    assert status == 'optimal' and add == 0 and drop.tolist() == [True]


def test_same_query_insertion_can_replace_one_original_at_cap():
    p = dict(selected_query=np.array([0]), selected_cost=np.array([[3.], [0.], [0.]]),
             deletion_gain=np.array([1.]), query=np.array([0]), cost=np.array([[-.5], [-.5], [-1.]]),
             gain=np.array([1.]), shortlist=np.zeros(1, bool), cap=np.array([1]), full_easy=25.)
    assert solve(p, 'insert_only')[1] == 'infeasible'
    assert solve(p, 'ordered_joint')[1] == 'optimal'
    assert solve(p, 'shortlist_joint')[1] == 'no_eligible_anchor'
