import numpy as np
import torch

from scripts.audit_m3w_cv_fixed_recipient_envelope import bound
from scripts.train_m3w_cv_fixed_recipient_support import paths, upper
from src.world_model import m3w_cv_fixed_recipient_support as model


def paths_x(x):
    y = np.zeros((len(x),12,2))
    y[:,:,0] = np.asarray(x)[:,None]
    return y


def test_projection_keeps_peak_and_excludes_undersized_donors():
    c,a = paths_x([0,0]),paths_x([4,4])
    b = np.stack((paths_x([1,-3]),paths_x([-3,1])),axis=1)
    p,l = model.project(c,a,b,.5)
    np.testing.assert_array_equal(l,[[True,False,True],[True,True,False]])
    np.testing.assert_allclose(model.peak(p-c[:,None])[l],2.)
    np.testing.assert_allclose(p[:,0],paths_x([2,2]))


def test_consensus_vetoes_pooled_favorite_if_one_source_disagrees():
    p = np.zeros((2,3,8))
    p[:,1,0] = 2.
    p[:,2,0] = 1.
    s = np.repeat(p[None],3,axis=0)
    s[1,0,1,4] = .01
    s[2,1,1,7] = -.01
    l = np.ones((2,3),bool)
    np.testing.assert_array_equal(model.choose(p,s,l,'pooled'),[1,1])
    np.testing.assert_array_equal(model.choose(p,s,l,'consensus'),[2,2])


def test_fixed_recipient_bound_dominates_any_causal_direction():
    c,o,y = paths_x([1,1]),paths_x([.7,.7]),paths_x([0,0])
    b = np.stack((paths_x([.5,.5]),paths_x([1.5,1.5])),axis=1)
    a,v = np.array([0,-1]),np.ones((2,12),bool)
    p = paths(c,b,a,np.array([2]),.5)
    np.testing.assert_array_equal((p!=c).any((1,2)),a>=0)
    q = np.array([0,0])
    z = upper(c,o,b,a,y,v,q)
    assert z[0]>0 and z[1]<0
    assert z[0] >= 100*(np.abs(c-y).sum()-np.abs(p-y).sum())/np.abs(c-y).sum()
    arbitrary = bound(c,o,paths_x([.5,1]),a>=0,y,v,q,.5)
    assert arbitrary[0]>=z[0] and arbitrary[1]>=z[1]-1e-12
    assert bound(c,o,paths_x([.5,1]),a>=0,y,v,q,None)[1]>arbitrary[1]


def test_finite_learning_targets_for_missing_future():
    c,o,u,p,y = (paths_x([z,z]) for z in (1.,.9,.75,1.25,0.))
    v = np.array([[True]*12,[False]*12])
    y[1] = np.nan
    z = model.targets(c,o,u,p,y,v,np.ones(2),2.)
    assert z.shape==(2,8)
    np.testing.assert_array_equal(z[:,7],[1,0])
    assert np.isfinite(z).all()
    net = torch.nn.Linear(3,8)
    assert torch.isfinite(model.objective(net,torch.zeros(2,3),torch.from_numpy(z)))
