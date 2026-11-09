import numpy as np

from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.world_model.m3w_av2_assignment_cost import choose, inputs, targets
from src.world_model.m3w_av2_recipient_reassignment import assignments, bank, render


def test_joint_ranker_changes_only_the_explicit_pair_term():
    opts = assignments(2)
    context, action = np.zeros((2, 3)), np.zeros((2, 2, 5))
    a, p = choose(context, action, np.array([0., 1.]), opts)
    np.testing.assert_array_equal(a, [0, 1])
    assert p[0] == p[1]
    np.testing.assert_array_equal(choose(context, action, np.zeros(2), opts)[0], [0, 0])


def test_pair_target_replays_exact_two_actor_nonadditivity_with_missing_steps():
    cv = np.zeros((2, 12, 2))
    a = cv.copy()
    a[0, :, 0], a[1, :, 1] = .5, 1.
    h = np.zeros((2, 8, 2))
    d = dict(cv=cv, ols4=cv, h=h)
    p, _ = bank(cv, h, a)
    opts = assignments(2)
    y = a.copy()
    v = np.ones((2, 12), dtype=bool)
    v[0, -2:] = False
    _, _, _, scale = inputs(d, p, opts, 2.)
    _, _, z = targets(d, p, opts, y, v, 2., scale)
    for i, (j, k) in enumerate(opts):
        left, right = cv.copy(), cv.copy()
        left[j], right[k] = p[j, 0], p[k, 1]
        full = render(cv, p, [j, k])
        g = pair_reduction(np.stack((full, left, right)), cv, y, v)[:, 1]
        np.testing.assert_allclose(scale*z[i], g[0]-g[1]-g[2], atol=1e-7)
