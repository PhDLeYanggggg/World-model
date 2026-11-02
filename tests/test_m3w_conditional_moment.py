import numpy as np
import pytest
import torch
from src.world_model import m3w_conditional_moment as api


def priors():
    return dict(probability=.25, fractions=[[.1,.2],[.2,.1]], references=[1.,2.], class_weights=[2/3,2.])


def test_decoder_is_coherent_and_has_matched_active_capacity():
    model=api.ConditionalCritic(462,64,17,priors())
    assert sum(p.numel() for p in model.parameters())==31381
    args=(torch.zeros(3,7,462),torch.zeros(3,7,5,dtype=torch.double),torch.ones(3,7,dtype=torch.double),
          torch.full((3,),-1),torch.tensor([0,1,2]),3)
    p=model.components(*args)['moments']
    assert torch.isfinite(p).all() and (p>=0).all()
    assert (p[:,:,0]<=p[:,:,2]).all() and (p[:,:,:2].sum(2)<=1+1e-12).all()
    assert (p[:,:,3]<=p[:,:,2]).all() and (p[:,:,4]<=p[:,:,1]).all()
    assert p[:,:,2].std(1).max()<1e-12 and p[:,:,3].std(1).max()<1e-12


def test_identified_loss_does_not_push_correct_mean_cost_toward_event_probability():
    q=torch.tensor(.25,dtype=torch.double,requires_grad=True)
    cond=torch.tensor([[[.2,0.,1.],[0.,.4,2.]]]*7,dtype=torch.double,requires_grad=True)
    probability=q.expand(4);conditional=cond[None].expand(4,-1,-1,-1)
    mean=api.factor_moments(probability,conditional)
    event=np.array([0.,0.,0.,1.]);truth=np.empty((4,7,5))
    for i,e in enumerate(event.astype(int)):
        b,h,r=cond.detach().numpy()[:,e].T
        truth[i]=np.column_stack((b,h,r,e*r,e*h))
    data=dict(causal=dict(scales=np.ones(4),allowed=np.ones((4,7),bool),initial=np.full(4,-1)),
        known=np.ones(4,bool),targets=truth,rms=np.ones(5),event=event,priors=priors())
    parts=dict(moments=mean,probability=probability,event_logit=torch.logit(probability),conditional=conditional)
    loss=api.losses(parts,data,np.arange(4),torch.zeros(4,dtype=torch.long),1,dict(rms=[1.,1.]),'identified')['total']
    loss.backward()
    assert abs(float(q.grad))<1e-12
    assert cond.grad.abs().max()<1e-12


def test_unknown_batch_has_no_target_loss_or_gradient():
    model=api.ConditionalCritic(4,8,17,priors())
    args=(torch.zeros(2,7,4),torch.zeros(2,7,5,dtype=torch.double),torch.ones(2,7,dtype=torch.double),torch.full((2,),-1),torch.tensor([0,1]),2)
    data=dict(causal=dict(scales=np.ones(2),allowed=np.ones((2,7),bool),initial=np.full(2,-1)),
        known=np.zeros(2,bool),targets=np.full((2,7,5),np.nan),rms=np.ones(5),event=np.full(2,np.nan),priors=priors())
    loss=api.losses(model.components(*args),data,np.arange(2),args[-2],2,dict(rms=[1.,1.]),'identified')
    loss['total'].backward()
    assert float(loss['total'].detach())==0 and loss['unknown_supervised_rows']==0
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in model.parameters())


def test_inference_rejects_realized_event_or_future_label_fields():
    with pytest.raises(ValueError,match='causal'):
        api.predict({},dict(event=np.ones(3)))


def test_preparation_uses_explicit_loss_event_and_does_not_infer_it_from_reference():
    y=np.full((3,7,5),np.nan)
    y[0]=[.1,.1,1.,0.,0.]
    y[1]=[0.,.3,0.,0.,.3]
    c=dict(scales=np.ones(3),envelope=np.ones((3,7)),allowed=np.ones((3,7),bool))
    data=dict(causal=c,known=np.array([True,True,False]),targets=y,input_hashes={})
    result=api.prepare(data,dict(a=dict(cv_ade=np.array([4.,1.,np.nan]))),dict(a=slice(0,3)),2.)
    np.testing.assert_array_equal(result['event'][:2],[0.,1.])
    assert np.isnan(result['event'][2])
    assert result['causal'] is c and 'event' not in c
