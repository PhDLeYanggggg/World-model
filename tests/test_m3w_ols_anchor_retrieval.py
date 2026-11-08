import numpy as np
import torch

from src.world_model.m3w_ols_anchor_retrieval import choose,network,past_features,score,supervision


def test_unobserved_future_is_harm_not_a_positive_or_near_tie():
    a = np.zeros((2,12,2)); b = a+1.; y = b.copy(); y[1] = np.nan
    mask = np.ones((2,12),bool); mask[1] = False
    z,g,valid,known = supervision(a,b,y,mask)
    np.testing.assert_array_equal(z,[2,0]); np.testing.assert_allclose(g,[np.sqrt(2),-np.sqrt(2)])
    assert valid.all() and known.tolist() == [True,False]


def test_translation_does_not_change_probe_or_candidates():
    h = np.arange(8,dtype=float)[None,:,None]**2*np.array([[[1.,.2]]])
    for a,b in zip(past_features(h),past_features(h+50.),strict=True):
        np.testing.assert_allclose(a,b,atol=1e-7)


def test_fixed_query_count_and_abstention_and_exact_feature_ablation():
    v = np.array([1.,2.,3.,5.]); legal = np.array([True,False,True,True])
    q = [np.arange(3),np.array([3])]
    np.testing.assert_array_equal(choose(v,legal,q),[False,False,True,False])
    assert not choose(v,legal,q,threshold=4.).any()
    x = torch.zeros(2,9); z = x.clone(); z[:,-5:] = 4.
    net = network(x.shape[1]); torch.testing.assert_close(score(net,z,'learned_motion'),score(net,x,'learned_probe'))
