import numpy as np
import pytest
from src.world_model.m3w_signed_bias_probe import groups_and_weights, objective, fit_intercept


def example():
    return np.array(['a','a','a','b','b']), np.array(['r']*5), np.array([1,1,2,1,1]), np.array([
        [True,True,False],[False,False,True],[True,True,False],[False,True,False],[True,False,True]])


def test_exact_bias_matches_quadratic_for_both_arms():
    sites, rec, frames, bank = example()
    p = np.zeros((5,2)); y=np.array([[1,2],[3,-1],[5,2],[1,-2],[4,1]],float)
    for aggregate in (False, True):
        d=fit_intercept(p,y,sites,rec,frames,bank,aggregate)
        assert d['fitting_objective_after']<=d['fitting_objective_before']
        np.testing.assert_allclose(d['fitting_loss_reduction'],d['expected_quadratic_reduction'])


def test_intercept_derivative_same_for_pointwise_and_aggregate():
    sites, rec, frames, bank = example();rng=np.random.default_rng(18)
    p,y=rng.normal(size=(2,5,2))
    a=fit_intercept(p,y,sites,rec,frames,bank,False)
    b=fit_intercept(p,y,sites,rec,frames,bank,True)
    assert a['unconstrained_offset']==b['unconstrained_offset']


def test_fitting_gradient_matches_independent_torch_autograd():
    import torch
    sites,rec,frames,bank=example();known=np.array([True,True,True,True,False])
    queries,weights=groups_and_weights(sites,rec,frames,known,bank)
    error=np.array([[1.,2],[3,4],[1,2],[2,3],[0,0]])
    for aggregate in (False,True):
        offset=torch.zeros(2,dtype=torch.float64,requires_grad=True);loss=offset.sum()*0
        for ix,parts,qw in queries:
            e=torch.tensor(error)-offset
            loss=loss+.5*qw*e[ix].square().mean()
            for part in parts:
                if len(part):loss=loss+qw/6*(e[part].mean(0).square().mean() if aggregate else e[part].square().mean())
        loss.backward()
        np.testing.assert_allclose(offset.grad.numpy(),-(error*weights[:,None]).sum(0))
        assert weights[-1]==0


def test_negative_offset_is_not_applied():
    sites,rec,frames,bank=example()
    d=fit_intercept(np.ones((5,2)),np.zeros((5,2)),sites,rec,frames,bank,True)
    assert d['nonnegative_offset']==[0.,0.] and d['fitting_loss_reduction']==0


def test_unknown_rows_do_not_change_fit():
    sites,rec,frames,bank=example();y=np.ones((5,2));y[-1]=np.nan
    p=np.zeros((5,2));a=fit_intercept(p,y,sites,rec,frames,bank,True)
    p[-1]=1000;b=fit_intercept(p,y,sites,rec,frames,bank,True)
    assert a==b and a['unknown_rows']==1


def test_empty_banks_retain_half_anchor_mass():
    sites,rec,frames,bank=example();bank[:]=False
    _,w=groups_and_weights(sites,rec,frames,np.ones(5,bool),bank)
    assert np.isclose(w.sum(),.5)


def test_source_equal_query_weight_not_row_equal_weight():
    sites,rec,frames,bank=example();bank[:]=False
    _,w=groups_and_weights(sites,rec,frames,np.ones(5,bool),bank)
    assert np.isclose(w[:3].sum(),.25) and np.isclose(w[3:].sum(),.25)
    assert w[2]==2*w[0]


def test_incomplete_source_rejected():
    sites,rec,frames,bank=example()
    with pytest.raises(ValueError):groups_and_weights(sites,rec,frames,np.array([True,True,True,False,False]),bank)
