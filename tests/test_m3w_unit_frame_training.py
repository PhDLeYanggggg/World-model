import numpy as np
import pytest
import torch

from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_unit_frame_training import ARMS, frame_loss, frame_prediction, fit_frame


def fixture():
    g = torch.Generator().manual_seed(11)
    batch = dict(geometry=torch.randn(12,476,generator=g),observed=torch.ones(12,8),
        baseline=torch.zeros(12,12,2),target=torch.randn(12,12,2,generator=g),
        valid=torch.ones(12,12,dtype=torch.bool),radius=torch.arange(1,13,dtype=torch.float32),
        rotation=torch.eye(2).expand(12,2,2).clone(),spatial_support=torch.ones(12,dtype=torch.bool))
    batch['spatial_support'][0]=False; batch['radius'][0]=0
    batch['valid'][1]=False; batch['target'][1]=float('nan')
    def selected(ids):
        return {k:v[ids] for k,v in batch.items()}
    return batch, selected


@pytest.mark.parametrize('arm', ARMS)
def test_no_future_inputs_no_anchor_fallback_and_valid_gradient(arm):
    batch,_=fixture(); torch.manual_seed(3); model=OfflineVisualForecast(476)
    with torch.no_grad():
        model.output.weight.fill_(.01)
    p=frame_prediction(model,batch,arm)
    torch.testing.assert_close(p[0],batch['baseline'][0],rtol=0,atol=0)
    changed=dict(batch,target=torch.full_like(batch['target'],300),valid=~batch['valid'])
    torch.testing.assert_close(p,frame_prediction(model,changed,arm),rtol=0,atol=0)
    loss,_=frame_loss(p,batch,arm)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)


@pytest.mark.parametrize('arm', ARMS)
def test_exact_phase_boundary_resume_and_unchanged_completed_state(tmp_path,arm):
    _,batch=fixture()
    config=dict(pretraining_updates=3,main_updates=4,batch_size=4,learning_rate=.0003,weight_decay=.0001,checkpoint_every=2)
    def run(path,stop=None):
        torch.manual_seed(17); model=OfflineVisualForecast(476)
        fit=fit_frame(model,batch,batch,[12,12],arm=arm,config=config,seed=17,identity={'test':'fixed'},
            checkpoint=path,heartbeat=lambda **_:None,stop_at=stop)
        return model,fit
    a,fit=run(tmp_path/'a.pt')
    run(tmp_path/'b.pt',3); b,other=run(tmp_path/'b.pt')
    for key,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
    assert fit['step']==other['step']==7
    assert other['new_updates_this_invocation']==4
    again,done=run(tmp_path/'b.pt')
    assert done['new_updates_this_invocation']==0
    for key,value in b.state_dict().items():
        torch.testing.assert_close(value,again.state_dict()[key],rtol=0,atol=0)
