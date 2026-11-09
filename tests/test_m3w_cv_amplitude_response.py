import numpy as np
import torch

from src.world_model import m3w_cv_amplitude_response as model


def slopes(x):
    return np.diff(x,axis=-2,prepend=np.zeros((*x.shape[:-2],1,5)))/np.diff((0.,*model.AMPLITUDES))[:,None]


def test_shape_constrained_response_has_correct_slopes():
    z = torch.randn(8,model.WIDTH,requires_grad=True)
    p,m,e = model.response(z,True)
    a = np.diff(slopes(p.detach().numpy()),axis=-2)
    assert (a[...,:2]<=1e-6).all()
    assert (a[...,2:]>=-1e-6).all()
    assert (p[...,2:4]>=0.).all()
    (p.sum()+m.sum()+e.sum()).backward()
    assert torch.isfinite(z.grad).all()


def test_label_curves_remain_convex_or_concave_with_missing_steps():
    rng = np.random.default_rng(91)
    c,o,p,y = rng.normal(size=(4,40,12,2))
    v = rng.random((40,12))>.5
    v[:4] = False
    y[~v] = np.nan
    z,m,e = model.targets(c,o,p,y,v,np.ones(40),2.)
    a = np.diff(slopes(z),axis=-2)
    assert (a[...,:2]<=1e-5).all()
    assert (a[...,2:]>=-1e-5).all()
    assert np.isfinite(z).all() and np.isfinite(m).all() and np.isfinite(e).all()


def test_equal_mass_uniform_control_preserves_query_recipients():
    cv = np.zeros((3,12,2))
    bank = np.stack((np.ones_like(cv),-2*np.ones_like(cv)),axis=1)
    a = np.array([0,2*model.M-1,-1])
    groups = [np.array([0,1]),np.array([2])]
    first = model.apply(cv,bank,a)
    second = model.uniform(cv,bank,a,groups)
    for q in groups:
        n1,n2 = (np.linalg.norm(p[q],axis=-1).max(1) for p in (first,second))
        np.testing.assert_array_equal(n1>0,n2>0)
        np.testing.assert_allclose(n1.sum(),n2.sum())
    assert (second[1]<0.).all()


def test_portfolio_can_offset_a_costly_action_and_keep_count_control():
    gains = np.array([[[5,5]],[[1,1]]],float)
    risk = np.array([[[1,1,0]],[[-1,-1,0]]],float)
    legal = np.ones((2,1),bool)
    groups = [np.arange(2)]
    a = model.portfolio(gains,risk,legal,np.ones((2,2)),groups,1.,0.)
    np.testing.assert_array_equal(a,[0,0])
    np.testing.assert_array_equal(model.independent(gains,risk,legal,groups,1.,match=a),a)
