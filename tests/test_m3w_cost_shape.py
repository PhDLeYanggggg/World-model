import inspect
import numpy as np
import pytest
from scipy.optimize import minimize
from src.world_model import m3w_cost_shape as s


def example(seed=3):
    rng = np.random.default_rng(seed); n = 120
    env = rng.uniform(.5,3,n); r,u = rng.uniform(0,1,(2,n))
    p = np.column_stack([env*.8,env*r,env*.2,env*r*u])
    y = p.copy(); y[:,1] = env*r**2; y[:,3] = y[:,1]*u**2
    sites = np.array(list('abc')*(n//3)); return p,y,env,sites


@pytest.mark.parametrize('preserve',[False,True])
def test_bounds_identity_columns_monotonicity_and_replay(preserve):
    p,y,e,sites = example(); m=s.fit(p,y,e,sites,'outer',preserve_mass=preserve)
    z=s.predict(m,p,e)
    np.testing.assert_array_equal(z[:,[0,2]],p[:,[0,2]])
    assert np.all(z[:,3] <= z[:,1]+1e-12) and np.all(z[:,1] <= e+1e-12)
    assert m == s.fit(p,y,e,sites,'outer',preserve_mass=preserve)
    assert all(np.min(np.diff(c['ordinates'])) >= -1e-12 for c in m['components'])
    if preserve: assert m['mass_preserved']
    assert 'target' not in inspect.signature(s.predict).parameters


@pytest.mark.parametrize('preserve',[False,True])
@pytest.mark.parametrize('seed',[1,9,21])
def test_convex_solver_agrees_with_independent_slsqp(preserve,seed):
    p,y,e,sites=example(seed); w=np.full(len(e),1/len(e)); r=p[:,1]/e
    fit=s.component(r,e,y[:,1],w,preserve); a=s.basis(r,e)
    f=lambda d: float(w @ (a@d-y[:,1])**2)
    constraints=[dict(type='eq',fun=lambda d:d.sum()-1)]
    target=float(w@y[:,1]); c=float(w@e)
    start=np.zeros(a.shape[1]); start[0]=target/c; start[-1]=1-target/c
    if preserve: constraints.append(dict(type='eq',fun=lambda d:float(w@(a@d))-target))
    other=minimize(f,start,method='SLSQP',bounds=[(0,1)]*a.shape[1],constraints=constraints,
                   options=dict(ftol=1e-12,maxiter=1000))
    assert other.success
    assert fit['MSE'] == pytest.approx(other.fun,abs=1e-8)
    assert fit['normalized_optimality_gap'] <= 1e-7


def test_zero_and_constant_scores_can_still_match_mass():
    p,y,e,sites=example(); p[:,[1,3]]=0
    m=s.fit(p,y,e,sites,'outer',preserve_mass=True)
    assert m['mass_preserved']
    y[:,[1,3]]=0; m=s.fit(p,y,e,sites,'outer',preserve_mass=True)
    np.testing.assert_array_equal(s.predict(m,p,e)[:,[1,3]],0)


def test_unknown_and_zero_envelope_rows_are_excluded_not_fitted():
    p,y,e,sites=example(); y[0]=np.nan; e[1]=0; p[1]=0; y[1]=0
    m=s.fit(p,y,e,sites,'outer',preserve_mass=True)
    assert m['fitting_rows']==118
    with pytest.raises(ValueError): s.fit(p,y,e,sites,'a',preserve_mass=True)
    y[0]=[np.nan,0,0,0]
    with pytest.raises(ValueError): s.fit(p,y,e,sites,'outer',preserve_mass=True)


def test_invalid_envelope_and_infeasible_target_rejected():
    p,y,e,sites=example(); y[:,1]=e+1
    with pytest.raises(ValueError): s.fit(p,y,e,sites,'outer',preserve_mass=True)
    p[0,3]=p[0,1]+1
    with pytest.raises(ValueError): s.predict(dict(),p,e)


def test_equal_locality_weights_not_row_frequency():
    p,y,e,sites=example(); take=np.r_[np.arange(120),np.flatnonzero(sites=='a')]
    a=s.fit(p,y,e,sites,'outer',preserve_mass=True)
    b=s.fit(p[take],y[take],e[take],sites[take],'outer',preserve_mass=True)
    np.testing.assert_allclose(s.predict(a,p,e),s.predict(b,p,e),atol=1e-9)
