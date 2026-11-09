import numpy as np

from scripts import run_m3w_av2_learned_reference as run


def test_support_preference_does_not_change_fixed_choice_count():
    z = np.zeros((4, 6))
    z[:, :2] = np.array([2., 1., 4., 3.])[:, None]
    z[:, 5] = [-1., 1., -1., -1.]
    opts = np.array([[0, 2], [1, 2], [0, 3], [1, 3]])
    point, order, conf = run.rank(z, opts, np.ones(4))
    assert point == 0
    np.testing.assert_array_equal(order, [1, 3])
    assert 0 < conf < 1


def test_predicted_excess_breaks_ties_without_future_labels():
    z = np.zeros((4, 6))
    z[:, :2] = np.array([3., 2., 2., 1.])[:, None]
    z[:, 5] = -1.
    z[0, 3] = 2.
    opts = np.array([[0, 2], [1, 2], [0, 3], [1, 3]])
    point, order, _ = run.rank(z, opts, np.ones(4))
    assert point == 0
    assert order[0] == 1
    assert set(order) == set(range(len(opts)))


def test_easy_head_uses_same_recipient_history_across_actions():
    z = run.torch.zeros((2, run.cost.WIDTH+1))
    z[1, 20:-1] = 100.
    result = run.Action()(z)
    run.torch.testing.assert_close(result[0, -1], result[1, -1], rtol=0., atol=0.)
