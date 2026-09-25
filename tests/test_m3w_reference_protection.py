import numpy as np
import pytest
import torch
from src.world_model.m3w_selected_risk_learning import EventMomentHead
from src.world_model.m3w_reference_protection import ContinuationHead,fit,loss_parts
from src.world_model.m3w_native_forecast import draw_batch
from tests.test_m3w_easy_harm_sampling import fixture


def data():
    x,y,s,e,_,pr=fixture(); torch.manual_seed(17); initial=EventMomentHead(6,5)
    target=np.where(pr['known'][:,None],y,0).astype(np.float32)
    rms=np.sqrt(np.sum(pr['weights'][:,None]*target.astype(float)**2,axis=0)).clip(1e-4)
    groups=[np.flatnonzero(pr['known'] & (s==site)) for site in sorted(set(s))]
    rng=torch.Generator().manual_seed(17); fixed=draw_batch(groups,8,rng)
    with torch.no_grad(): loss,_=loss_parts(initial(torch.tensor(x[fixed],dtype=torch.float32),
        torch.tensor(e[fixed],dtype=torch.float32)),torch.tensor(target[fixed]),torch.tensor(rms,dtype=torch.float32))
    control=dict(preprocess=pr,loss_scales=rms,fixed_ids=fixed,sampler_rng=rng.get_state(),trace=[dict(moment_mse=float(loss))])
    return x,y,s,e,pr,initial,control


def test_protected_initial_equal_and_reference_gradients_absent():
    *_,initial,_=data(); x=torch.randn(7,6); env=torch.ones(7)
    model=ContinuationHead(initial,'protected')
    assert torch.equal(model(x,env),initial(x,env))
    model(x,env).sum().backward()
    assert all(p.grad is None for p in model.reference.parameters())
    assert torch.equal(model.learned.network[-1].weight.grad[[0,2]],torch.zeros(2,5))


@pytest.mark.parametrize('arm',['continued','protected'])
def test_continuation_resume_exact(tmp_path,arm):
    args=data(); settings=dict(width=5,steps=6,batch_size=8,learning_rate=.001,
        gradient_clip=5.,checkpoint_every=3,heartbeat_every=3)
    common=dict(arm=arm,seed=17,settings=settings,identity={'test':True},heartbeat=lambda **k:None)
    full,_=fit(*args,directory=tmp_path/'full',**common)
    fit(*args,directory=tmp_path/'resume',stop_at=3,**common)
    resumed,m=fit(*args,directory=tmp_path/'resume',resume=True,**common)
    for k,v in full.state_dict().items(): assert torch.equal(v,resumed.state_dict()[k])
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=False)
    np.testing.assert_array_equal(a['draws'],b['draws']); assert torch.equal(a['sampler_rng'],b['sampler_rng'])
    assert m['unknown_training_draws']==0
    if arm=='protected':
        x=torch.tensor(args[0],dtype=torch.float32); e=torch.tensor(args[3],dtype=torch.float32)
        assert torch.equal(full(x,e)[:,[0,2]],args[-2](x,e)[:,[0,2]])


def test_invalid_arm_and_loss_rejected():
    with pytest.raises(ValueError): ContinuationHead(EventMomentHead(3,5),'chosen_on_test')
    with pytest.raises(ValueError): loss_parts(torch.ones(3,4),torch.ones(3,4),torch.zeros(4))
