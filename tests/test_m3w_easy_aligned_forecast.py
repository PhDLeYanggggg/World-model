import torch

from src.world_model.m3w_easy_aligned_forecast import objective


def test_only_observed_easy_above_cv_receives_an_extra_gradient():
    p=torch.zeros(4,12,2);y=p.clone();c=p.clone();m=torch.ones(4,12,dtype=torch.bool)
    y[:,:,0]=torch.tensor([1.,3.,1.,1.])[:,None];c[2,:,0]=-.5;p[2,:,0]=.5
    m[3]=False;c[3,:,0]=10.
    p.requires_grad_();w=torch.ones(4)
    a,b,h=objective(p,y,m,w,c,2.,aligned=False)
    c[0,:,0]=.5
    d,e,j=objective(p,y,m,w,c,2.,aligned=True)
    assert a==b==e and j==.125 and d-a==.25
    d.backward(retain_graph=True); grad=p.grad.clone();p.grad.zero_();a.backward()
    torch.testing.assert_close(grad[1:],p.grad[1:]);assert not torch.equal(grad[0],p.grad[0])
