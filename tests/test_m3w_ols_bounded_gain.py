import pytest
import torch

from src.world_model.m3w_ols_bounded_gain import fractional_target,objective,score,network,logits


def test_fractional_labels_scale_and_missing_gain():
    d = torch.tensor([[.2,.8,0.]])
    g,h = torch.tensor([[.1,-.8,0.]]),torch.tensor([[0.,.8,0.]])
    a,m = fractional_target(g,h,d)
    b,n = fractional_target(3*g,3*h,3*d)
    torch.testing.assert_close(a,b)
    assert m.tolist() == n.tolist() == [[True,True,False]]
    assert a[0,1].tolist() == [0.,1.]
    with pytest.raises(ValueError,match='geometric'):
        fractional_target(g+1.,h,d)


def test_score_is_bounded_and_invalid_actions_do_not_train():
    z = torch.tensor([[[99.,-99.],[-99.,99.],[0.,0.]]],requires_grad=True)
    d = torch.tensor([[1.,.4,0.]])
    t,m = fractional_target(torch.zeros_like(d),torch.zeros_like(d),d)
    objective(z,t,m).backward()
    assert z.grad[0,2].abs().sum() == 0.
    torch.testing.assert_close(score(z,d),torch.tensor([[1.,-.8,0.]]))


def test_paired_ablations_hide_only_the_declared_features():
    x = torch.zeros(2,12,21); n = torch.zeros(2,6)
    torch.manual_seed(1)
    net = network(27)
    q = x.clone(); q[..., -5:-1] = 5.
    torch.testing.assert_close(logits(net,q,n,'bounded_no_hindcast'),logits(net,x,n,'bounded_full'))
    torch.testing.assert_close(logits(net,x,n+5.,'bounded_no_neighbors'),logits(net,x,n,'bounded_full'))
