import numpy as np
import pytest

from src.evaluation.m3w_candidate_headroom import segment_oracle, budget_envelope


def path(x, y=0., n=1):
    return np.tile([x, y], (n, 12, 1)).astype(float)


def test_segment_oracle_finds_interior_and_keeps_perfect_floor():
    base, candidate, target = path(0., n=2), path(2., n=2), path(1., n=2)
    target[1] = 0
    out = segment_oracle(base, candidate, target)
    np.testing.assert_allclose(out['alpha'], [.5, 0.], atol=1e-9)
    np.testing.assert_allclose(out['feasible_loss'], [0., 0.], atol=1e-9)
    np.testing.assert_allclose(out['binary_loss'], [1., 0.])
    assert np.all(out['lower_loss'] <= out['feasible_loss'])


def test_endpoint_and_zero_delta_cases():
    out = segment_oracle(path(0.), path(1.), path(3.))
    assert out['alpha'][0] == 1 and out['feasible_loss'][0] == 2
    out = segment_oracle(path(0.), path(0.), path(3.))
    assert out['alpha'][0] == 0 and out['feasible_loss'][0] == 3
    out = segment_oracle(path(0.), path(-1.), path(3.))
    assert out['alpha'][0] == 0


def test_ade_uses_one_alpha_for_entire_path_not_each_waypoint():
    base, cand, target = path(0.), path(2.), path(0.)
    target[:, :6] = [2., 0.]
    out = segment_oracle(base, cand, target)
    assert out['feasible_loss'][0] == 1.


def test_random_solution_brackets_convex_dense_grid():
    rng = np.random.default_rng(29)
    base, cand, target = [rng.normal(size=(11, 12, 2)) for _ in range(3)]
    out = segment_oracle(base, cand, target, iterations=36)
    grid = np.linspace(0, 1, 2001)
    loss = np.linalg.norm(base[:, None]+grid[None, :, None, None]*(cand-base)[:, None]-target[:, None], axis=-1).mean(-1)
    brute = loss.min(1)
    assert np.all(out['lower_loss'] <= brute+1e-12)
    assert np.all(out['feasible_loss'] <= brute+1e-12)
    assert np.max(out['feasible_loss']-out['lower_loss']) < 1e-8
    assert np.all(out['feasible_loss'] <= out['binary_loss'])


def test_segment_pool_is_not_a_bound_for_arbitrary_convex_mixtures():
    base, target = path(0.), path(1.)
    a, b = path(1., 1.), path(1., -1.)
    assert np.allclose((a+b)/2, target)
    assert segment_oracle(base, a, target)['feasible_loss'][0] > .7
    assert segment_oracle(base, b, target)['feasible_loss'][0] > .7


def test_budget_envelope_scene_aggregation_and_no_harm():
    ref = np.array([10., 10., 1., 1.])
    upper = np.array([8., 10., 0., 1.])
    lower = upper-.01
    scenes = np.array([0, 0, 1, 1])
    zero = budget_envelope(ref, upper, lower, scenes, 0.)
    assert zero['feasible_gain_percent'] == zero['optimistic_gain_percent'] == 0.
    half = budget_envelope(ref, upper, lower, scenes, .5)
    assert half['allowed_rows'] == 2
    assert np.isclose(half['feasible_gain_percent'], 100*1.5/11)
    assert half['optimistic_gain_percent'] >= half['feasible_gain_percent']
    assert budget_envelope(ref, ref, ref, scenes, 1.)['feasible_gain_percent'] == 0.


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        segment_oracle(path(0.), path(1.)[:, :-1], path(2.))
    with pytest.raises(ValueError):
        segment_oracle(path(0.), path(np.nan), path(2.))
    with pytest.raises(ValueError):
        budget_envelope(np.ones(3), np.zeros(3), np.zeros(3), [0, 0, 0], 1.1)
