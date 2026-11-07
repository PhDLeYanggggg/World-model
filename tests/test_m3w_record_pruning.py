import numpy as np

from src.world_model.m3w_record_pruning import choose, observe, oracle, realized


def test_prune_at_most_one_and_never_look_ahead():
    g = [np.arange(2), np.arange(2, 4)]
    z = choose(g, np.array([.2, .1, 100., 200.]), .05)
    np.testing.assert_array_equal(z, [True, False, False, True])
    np.testing.assert_array_equal(choose(g[:1], np.array([.2, .1]), .05), z[:2])


def test_oracle_retains_support_and_drops_a_harmful_action():
    z = dict(ref_easy=np.array([20., 0., 0.]), all_excess=np.array([-.4, 3., -.4]),
             easy_excess=np.array([-.4, 0., 0.]), easy_signed=np.zeros(3),
             remove_lower=np.array([-1., 3., -1.]))
    drop, status = oracle(z, np.array([0, 1, 1]), np.ones(3), 20., True)
    assert status == 'optimal'
    np.testing.assert_array_equal(drop, [False, True, False])
    assert observe(z, drop, 20.) == (True, False)
    z['ref_easy'][:] = 0
    assert oracle(z, np.arange(3), np.ones(3), 0., False) == (None, 'no_original_support')


def test_missing_outcome_penalizes_removing_as_well_as_adding():
    b = np.zeros((2, 12, 2))
    a = b.copy()
    a[:, :, 0] = 2.
    valid = np.ones((2, 12), bool)
    valid[0] = False
    m = realized(b, a, b, valid, 1.)
    np.testing.assert_array_equal(m['remove_lower'], [-2., 2.])
