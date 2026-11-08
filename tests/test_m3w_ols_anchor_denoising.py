import numpy as np
import torch

from src.world_model.m3w_ols_anchor_denoising import design,loss,network


def test_constant_velocity_needs_no_anchor_or_slope_adjustment():
    h = np.arange(8,dtype=float)[None,:,None]*np.array([[[1.,2.]]])
    x,p,a,b,s,m = design(h)
    np.testing.assert_allclose(a,0.)
    np.testing.assert_allclose(b,0.)
    np.testing.assert_allclose(p['raw_line4'],p['ols4'])
    np.testing.assert_array_equal(x,design(h+100.)[0])


def test_shift_and_slope_have_same_maximum_displacement_on_joint_support():
    h = np.arange(8,dtype=float)[None,:,None]**2*np.array([[[1.,.2]]])
    _,p,a,b,_,ok = design(h)
    assert ok.all()
    np.testing.assert_allclose(np.linalg.norm(a,axis=2).max(1),np.linalg.norm(b,axis=2).max(1))
    np.testing.assert_allclose(a[:,2],a[:,-1])
    np.testing.assert_allclose(p['ols4'][:,1],2*p['ols4'][:,0])


def test_hidden_coordinates_do_not_change_training_signal():
    net = network(3); x = torch.ones(2,3)
    p = torch.zeros(2,12,2); d = torch.ones_like(p)
    mask = torch.zeros(2,12,dtype=torch.bool); mask[0,:2] = True
    y = torch.ones_like(p)
    a = loss(net,x,p,d,y,mask,torch.ones(2))
    y[~mask] = torch.nan
    b = loss(net,x,p,d,y,mask,torch.ones(2))
    torch.testing.assert_close(a,b)
    assert all(torch.isfinite(v).all() for v in torch.autograd.grad(b,net.parameters()))
