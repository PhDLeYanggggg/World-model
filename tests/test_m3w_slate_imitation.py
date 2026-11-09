import numpy as np
import torch

from src.world_model import m3w_slate_imitation as m


def test_slate_enumerates_abstention_and_only_distinct_pairs():
    r = np.array([1,1,2,3])
    x = m.options(r,2)
    np.testing.assert_array_equal(x[0],[-1,-1])
    assert len(x) == 10
    for a,b in x[(x>=0).all(1)]:
        assert r[a]!=r[b]
    assert len(m.options(r,1))==5


def test_pair_score_is_permutation_invariant_and_null_has_zero_score():
    net = m.Slate()
    a = torch.randn(6,2,45)
    g = torch.randn(6,4)
    on = torch.ones(6,2,dtype=torch.bool)
    on[0] = False
    on[1,1] = False
    for x,y in zip(net(a,g,on),net(a.flip(1),g,on.flip(1)),strict=True):
        torch.testing.assert_close(x,y)
        assert x[0].item()==0
    u,v = net(a,g,on)
    assert u[1].item()==v[1].item()


def test_matched_control_keeps_exact_radii_without_growing_bank():
    cv = np.zeros((3,12,2))
    h = np.zeros((3,8,2))
    h[...,0] = np.arange(8)
    bank = np.zeros((3,2,12,2))
    bank[...,0] = np.array([[3.,5.],[6.,8.],[8.,7.]])[:,:,None]
    ind,uniform = m.match(m.Slate(),h,cv,cv,bank,[4.,7.])
    for p in (ind,uniform):
        peak = np.linalg.norm(p-cv,axis=-1).max(1)
        np.testing.assert_allclose(np.sort(peak),[0.,4.,7.])
