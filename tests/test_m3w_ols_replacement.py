import numpy as np
import torch

from src.world_model.m3w_ols_replacement import choose, features, targets, network, prediction


def test_translation_invariant_features_and_constant_velocity():
    x = np.zeros((1,8,2)); x[0,:,0] = np.arange(8)
    a, cv, ols, paths, scale = features(x)
    b, *_ = features(x+1000.)
    np.testing.assert_allclose(a,b,atol=1e-9)
    np.testing.assert_allclose(cv,ols,atol=1e-9)
    np.testing.assert_allclose(paths[:,0],ols,atol=1e-9)
    assert a.shape[:2] == (1,4) and scale.item() == 1.
    assert prediction(network(a.shape[-1]),torch.from_numpy(a)).shape == (1,4,2)


def test_unknown_targets_have_only_negative_disagreement_credit():
    h = np.arange(16,dtype=float).reshape(1,8,2)
    _,_,ols,paths,s = features(h)
    valid = np.zeros((1,12),bool)
    a = targets(ols,paths,np.full_like(ols,np.nan),valid,s)
    b = targets(ols,paths,ols,valid,s)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(a[...,0],-np.linalg.norm(paths-ols[:,None],axis=3).max(2)/(12*s[:,None]))
    np.testing.assert_array_equal(a[...,1],-a[...,0])


def test_same_candidate_support_for_query_budget():
    a = np.array([[3.,2.],[4.,5.],[2.,1.],[9.,2.]])
    legal = np.array([[True,True],[False,False],[True,False],[True,True]])
    q = [np.arange(3),np.array([3])]
    np.testing.assert_array_equal(choose(a,legal,q),[0,-1,0,-1])
    assert (choose(-a,legal,q) >= 0).sum() == 2
