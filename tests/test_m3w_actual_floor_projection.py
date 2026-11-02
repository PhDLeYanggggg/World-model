import numpy as np
import pytest
from scipy.optimize import minimize

from src.evaluation.m3w_actual_floor_projection import project


def example(seed=7,n=9):
    rng = np.random.default_rng(seed)
    p = rng.uniform(.01,20,(n,7,5))
    p[:,:,0] = np.minimum(p[:,:,0],p[:,:,2])
    p[:,:,3] = np.minimum(p[:,:,3],p[:,:,2])
    p[:,:,4] = np.minimum(p[:,:,4],p[:,:,1])
    return p,rng.uniform(0,15,(n,7)),rng.uniform(.5,3,5)


def test_projection_matches_independent_constrained_optimizer():
    p,d,r = example(n=3)
    q = project(p,d,r)
    for i,k in np.ndindex(d.shape):
        original = p[i,k,[0,1,4]]; radius = d[i,k]; variance = r[[0,1,4]]**2
        fn = lambda x:np.sum((x-original)**2/variance)
        solved = minimize(fn,np.zeros(3),method='SLSQP',bounds=[(0,None)]*3,
            constraints=[{'type':'ineq','fun':lambda x:radius-x[0]-x[1]},
                         {'type':'ineq','fun':lambda x:x[1]-x[2]}],
            options={'ftol':1e-11,'maxiter':1000})
        assert solved.success
        np.testing.assert_allclose(q[i,k,[0,1,4]],solved.x,rtol=1e-5,atol=2e-5)


def test_feasible_cost_error_does_not_increase_but_gain_loss_not_guaranteed():
    p,d,r = example(n=200)
    q = project(p,d,r)
    rng = np.random.default_rng(4)
    y = p.copy(); y[:,:,0] = 0
    y[:,:,1] = rng.uniform(size=d.shape)*d
    y[:,:,4] = y[:,:,1]*rng.uniform(size=d.shape)
    before = np.square((p-y)/r).sum(-1)
    after = np.square((q-y)/r).sum(-1)
    assert np.all(after <= before+1e-10)
    assert np.all(q[:,:,0]+q[:,:,1] <= d+1e-12)
    assert np.all(q[:,:,4] <= q[:,:,1])
    np.testing.assert_array_equal(q[:,:,2:4],p[:,:,2:4])
    np.testing.assert_allclose(project(q,d,r),q,rtol=0,atol=1e-12)


def test_zero_radius_and_already_feasible_prediction():
    p = np.ones((1,7,5)); p[:,:,4] = .5
    q = project(p,np.zeros((1,7)),np.ones(5))
    assert not q[:,:,[0,1,4]].any()
    np.testing.assert_array_equal(q[:,:,2:4],p[:,:,2:4])
    np.testing.assert_array_equal(project(p,np.full((1,7),3.),np.ones(5)),p)


def test_invalid_costs_or_label_shaped_radius_rejected():
    p,d,r = example()
    with pytest.raises(ValueError):
        project(p,np.full_like(p,np.nan),r)
    p[:,:,4] = p[:,:,1]+1
    with pytest.raises(ValueError):
        project(p,d,r)
