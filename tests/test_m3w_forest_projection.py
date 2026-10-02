import numpy as np
import pytest

from src.world_model import m3w_forest_projection as api
from src.world_model.m3w_source_forest import project_moments


def test_proportional_projection_can_create_risk_eligibility():
    raw = np.array([[20., 1., 40., 40., 1.]])
    env = np.array([2.])
    old = project_moments(raw, env)
    repaired = api.harm_first(raw, env)
    np.testing.assert_allclose(old[0, :2], [40/21, 2/21])
    np.testing.assert_array_equal(repaired, [[1., 1., 40., 40., 1.]])
    assert api.eligible(old, np.ones(1, bool), np.ones(1, bool))[0]
    assert not api.eligible(repaired, np.ones(1, bool), np.ones(1, bool))[0]


def test_random_monotonicity_and_feasibility():
    rng = np.random.default_rng(43)
    raw = rng.lognormal(size=(2000, 5))
    env = rng.lognormal(size=2000)
    old, new = project_moments(raw, env), api.harm_first(raw, env)
    assert (new[:, 0] <= old[:, 0]+1e-12).all()
    assert (new[:, 1] >= old[:, 1]-1e-12).all()
    assert (new[:, 4] >= old[:, 4]-1e-12).all()
    np.testing.assert_array_equal(new[:, 2:4], old[:, 2:4])
    assert (new[:, :2].sum(1) <= env+1e-12).all()
    assert (new[:, 0] <= new[:, 2]).all()
    assert (new[:, 3] <= new[:, 2]).all()
    assert (new[:, 4] <= new[:, 1]).all()
    take = np.ones(len(env), bool)
    assert not (api.eligible(new, take, take) & ~api.eligible(old, take, take)).any()


def test_nonbinding_projection_and_input_immutability():
    raw = np.array([[1., 2., 4., 3., .5], [0., 0., 1., 1., 0.]])
    original = raw.copy()
    env = np.array([4., 0.])
    np.testing.assert_array_equal(api.harm_first(raw, env), project_moments(raw, env))
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize('raw,env', [([[np.nan]*5], [1.]), ([[-1., 1., 1., 1., 1.]], [1.]),
                                  ([[1.]*5], [-1.]), ([[1.]*4], [1.])])
def test_rejects_invalid_inputs(raw, env):
    with pytest.raises(ValueError):
        api.harm_first(np.array(raw), np.array(env))


def test_unknown_labels_do_not_change_actions():
    p = np.array([[10., .01, 20., 20., .01], [1., 3., 4., 4., 3.]])
    m = np.ones(2, bool)
    take = api.eligible(api.harm_first(p, np.array([12., 4.])), m, m)
    y = np.array([[np.nan]*5, [0., 3., 4., 4., 3.]])
    result = api.bounds(y, take, np.array([12., 4.]))
    np.testing.assert_array_equal(take, [True, False])
    assert result['selected_unknown'] == 1
    assert not result['finite_completion_supported']


def test_query_matched_control_counts_and_unknown_penalty():
    y = np.array([[5., 0., 10., 10., 0.], [0., 3., 10., 10., 3.], [np.nan]*5])
    old = np.array([True, True, True])
    new = np.array([True, False, False])
    # Two known rows share a query; the unknown is in another query with zero retention.
    rec, frame = np.array(['a', 'a', 'b']), np.array([1, 1, 1])
    assert api.matched_utility(y, old, new, np.array([5., 3., 7.]), rec, frame) == 4.


def test_empty_actions_not_pass():
    y = np.array([[1., 0., 2., 2., 0.]])
    result = api.bounds(y, np.zeros(1, bool), np.ones(1))
    assert result['easy_selected_risk_upper'] is None
    assert not result['finite_completion_supported']


def test_locality_bootstrap_averages_heads_before_resampling():
    result = api.interval([('a', 1.), ('a', 3.), ('b', 4.)], 3000, 43)
    assert result['mean'] == 3.
    assert result['localities'] == 2
    assert result['CI95'] == [2., 4.]
