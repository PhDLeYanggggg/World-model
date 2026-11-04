import numpy as np
import pytest
import torch

from src.world_model import m3w_convex_amplitude_cost as api
from tests.test_m3w_amplitude_cost import packet


def test_same_parameters_initial_raw_state_and_unconstrained_control_are_exact():
    bias=np.array([-.4,-.8,.1,-.2,.3],np.float32)
    old=api.OriginalHead(428,64,17,bias,'context')
    new=api.ShapeHead(428,64,17,bias,'context',shape_arm='convex')
    control=api.ShapeHead(428,64,17,bias,'context',shape_arm='unconstrained')
    assert sum(p.numel()for p in new.parameters())==28229
    api.parent.base.base.exact(old.state_dict(),new.state_dict())
    x=torch.randn(5,7,428);env=torch.arange(1.,6.)[:,None]*torch.tensor(api.ALPHA)
    segment=torch.tensor([0,0,1,1,1])
    torch.testing.assert_close(old(x,env,segment,2),control(x,env,segment,2),rtol=0,atol=0)
    torch.testing.assert_close(old(x,env,segment,2)[:,6],new(x,env,segment,2)[:,6],rtol=1e-6,atol=1e-6)


def test_random_fitted_heads_have_convex_nested_harm_and_coherent_gradients():
    model=api.ShapeHead(12,8,29,np.zeros(5,np.float32),'context',shape_arm='convex')
    with torch.no_grad():
        for p in model.parameters():p.normal_()
    x=torch.randn(10,7,12);env=torch.arange(0.,10.)[:,None]*torch.tensor(api.ALPHA)
    out=model(x,env,torch.tensor([0]*5+[1]*5),2)
    assert torch.isfinite(out).all()and (out>=0).all()
    assert (out[:,:,0]+out[:,:,1]<=env+1e-5).all()
    assert (out[:,:,4]<=out[:,:,1]+1e-5).all()
    assert not out[:,0,[0,1,4]].count_nonzero()
    for harm in (out[:,:,1],out[:,:,4],out[:,:,1]-out[:,:,4]):
        slopes=torch.diff(harm,dim=1)/torch.tensor(np.diff(api.ALPHA))
        assert (torch.diff(slopes,dim=1)>=-2e-5).all()
    out.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters())
    model.eval()
    stable=model(x,env,torch.tensor([0]*5+[1]*5),2)
    assert stable.dtype==torch.float64
    for harm in (stable[:,:,1],stable[:,:,4],stable[:,:,1]-stable[:,:,4]):
        slopes=torch.diff(harm,dim=1)/torch.tensor(np.diff(api.ALPHA))
        assert (torch.diff(slopes,dim=1)>=-1e-10).all()


def test_convex_training_exact_resume_and_rejects_future_inputs(tmp_path):
    inputs,target,valid,pr=packet()
    data=api.parent.training_data(api.parent.causal_bank(**inputs),target,valid,1.,pr)
    settings=dict(width=8,learning_rate=.001,query_batch_size=2,minimum_updates=4,largest_source_cycles=1,
        gradient_clip=5.,checkpoint_every=1,heartbeat_every=1)
    kwargs=dict(seed=17,settings=settings,identity=dict(synthetic=True))
    a=api.fit(data,path=tmp_path/'a.pt.gz',**kwargs)
    api.fit(data,path=tmp_path/'b.pt.gz',stop_at=2,**kwargs)
    b=api.fit(data,path=tmp_path/'b.pt.gz',resume=True,**kwargs)
    api.parent.base.base.exact({k:v for k,v in a.items()if k!='seconds'},
        {k:v for k,v in b.items()if k!='seconds'})
    np.testing.assert_array_equal(api.predict(a,data['causal']),api.predict(b,data['causal']))
    assert api.parent.base.base.CostHead is api.OriginalHead
    with pytest.raises(ValueError):api.predict(a,dict(data['causal'],future_target=target))
