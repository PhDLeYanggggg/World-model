import numpy as np
from src.world_model.m3w_incumbent_relative import targets, choices, replay


def test_relative_cost_difference_matches_actual_override_for_random_rows():
    rng = np.random.default_rng(92)
    floor, neural = rng.exponential(3, (2, 1000))
    old = rng.random(1000) > .5; cv = floor+1
    u, r = targets(cv, floor, neural, old, arm='incumbent_reference', event='all', easy_cut=2)
    incumbent = np.where(old, neural, floor); alternate = np.where(old, floor, neural)
    np.testing.assert_allclose(u[:, 0]-u[:, 1], incumbent-alternate)
    np.testing.assert_allclose(r[:, 0], incumbent)
    np.testing.assert_allclose(r[:, 1], np.maximum(alternate-incumbent, 0))
    assert np.all((u[:, 0] == 0) | (u[:, 1] == 0))


def test_vector_scalar_choices_and_direction_invariants():
    rng = np.random.default_rng(93)
    moving = rng.random(1000) > .2; old = (rng.random(1000) > .5) & moving
    u = rng.uniform(0, 1, (1000, 2)); r = rng.uniform(0, 1, (1000, 2)); r[:, 1] *= .03
    for direction in ('both', 'add', 'remove'):
        new = choices(u, r, moving, old, arm='incumbent_reference', direction=direction)
        np.testing.assert_array_equal(new, replay(u, r, moving, old, arm='incumbent_reference', direction=direction))
        assert not new[~moving].any()
        if direction == 'add': assert new[old].all()
        if direction == 'remove': assert not new[~old].any()
    floor_choice = choices(u, r, moving, old, arm='floor_reference')
    np.testing.assert_array_equal(floor_choice, replay(u, r, moving, old, arm='floor_reference'))


def test_unsupported_cost_labels_do_not_change_causal_decisions():
    old = np.array([True, False]); moving = np.ones(2, bool)
    u = np.array([[1., 0.], [1., 0.]]); r = np.array([[1., .01], [1., .01]])
    before = choices(u, r, moving, old, arm='incumbent_reference')
    for errors in (np.full(2, np.nan), np.array([1., 100000.])):
        targets(errors, errors, errors, old, arm='incumbent_reference', event='all', easy_cut=2)
        np.testing.assert_array_equal(before, choices(u, r, moving, old, arm='incumbent_reference'))
