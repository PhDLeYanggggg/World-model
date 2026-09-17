import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from src.evaluation.m3w_conditional_geometric_median import (
    geometric_median, forest_training_weights, conditional_ade_prediction,
)


def test_zero_atom_can_be_optimal_without_majority_mass():
    y = np.array([[0.,0.],[-1.,0.],[1.,0.]])
    point, info = geometric_median(y, [.2,.4,.4])
    np.testing.assert_array_equal(point, [0.,0.])
    assert info['zero_optimal'] and info['gap_bound']==0


def test_iteration_leaves_zero_when_directional_weight_exceeds_zero_atom():
    point, info = geometric_median([[0.,0.],[2.,0.]], [.4,.6])
    np.testing.assert_allclose(point, [2.,0.], atol=1e-8)
    assert info['converged'] and not info['zero_optimal']
    assert info['objective']<=.80000001


def test_equivariance_and_triangle_solution_not_coordinatewise_median():
    y = np.array([[0.,0.],[2.,0.],[1.,np.sqrt(3.)]])
    expected = y.mean(0)
    p, info = geometric_median(y, np.ones(3))
    np.testing.assert_allclose(p, expected, atol=1e-7)
    r = np.array([[0.,-1.],[1.,0.]])
    shifted, _ = geometric_median(3*y@r+[4,-2], np.ones(3))
    np.testing.assert_allclose(shifted, 3*p@r+[4,-2], atol=2e-6)
    assert info['converged']


def test_iteration_limit_is_visible_and_invalid_weights_rejected():
    _, info = geometric_median([[0.,0.],[2.,0.]], [.4,.6], max_iterations=1)
    assert not info['converged'] and info['gap_bound']>0
    with pytest.raises(ValueError):
        geometric_median([[0.,0.]], [-1.])


def test_exact_forest_mean_replay_and_bootstrap_rejection():
    rng = np.random.default_rng(19)
    x, y, q = rng.normal(size=(50,3)), rng.normal(size=(50,24)), rng.normal(size=(11,3))
    forest = ExtraTreesRegressor(n_estimators=9, min_samples_leaf=3, random_state=17).fit(x,y)
    w = forest_training_weights(forest,x,q)
    np.testing.assert_allclose(w@y, forest.predict(q), atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        forest_training_weights(forest, x[:-1], q)
    forest.set_params(bootstrap=True)
    with pytest.raises(ValueError):
        forest_training_weights(forest,x,q)


def test_training_only_risk_not_above_mean_or_zero():
    rng = np.random.default_rng(29)
    target = rng.normal(size=(13,12,2))
    target[:8] = 0
    weights = np.ones((2,13))/13
    weights[1] = np.arange(1,14)/np.arange(1,14).sum()
    p, details = conditional_ade_prediction(target,weights)
    for i in range(2):
        risk = np.einsum('nt,n->', np.linalg.norm(p[i]-target,axis=-1), weights[i])/12
        mean = np.einsum('n,ntd->td', weights[i], target)
        mean_risk = np.einsum('nt,n->', np.linalg.norm(mean-target,axis=-1), weights[i])/12
        zero_risk = np.einsum('nt,n->', np.linalg.norm(target,axis=-1), weights[i])/12
        assert risk<=min(mean_risk,zero_risk)+1e-7
        assert all(d['converged'] for d in details[i])
