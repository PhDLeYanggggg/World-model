import numpy as np
import pytest
import torch
from src.world_model.m3w_source_temporal_centered import ARMS, CenteredTemporalDynamics, center_observed_tokens
from src.world_model.m3w_source_pretrained_temporal import TemporalSourceDynamics
from src.world_model.m3w_source_motion_candidate import fit_motion_candidate

torch.set_num_threads(4)


def test_common_appearance_and_batch_composition_invariance():
    torch.manual_seed(2)
    x = torch.randn(3,8,512); cov = torch.ones(3,8)
    for mode in ARMS:
        y = center_observed_tokens(x,cov,mode)
        torch.testing.assert_close(y,center_observed_tokens(x+torch.randn(3,1,512),cov,mode),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(y[:1],center_observed_tokens(x[:1],cov[:1],mode),atol=0,rtol=0)
        torch.testing.assert_close(y.mean(1),torch.zeros(3,512),atol=1e-6,rtol=0)


def test_unit_scale_constant_missing_and_unobserved_tokens():
    x = torch.randn(2,8,512); cov=torch.ones(2,8); cov[0,0]=0; cov[1]=0
    a=center_observed_tokens(x,cov,'centered_unit')
    torch.testing.assert_close(a,center_observed_tokens(x*3,cov,'centered_unit'))
    x[0,0]=1000; x[1]=2000
    torch.testing.assert_close(a,center_observed_tokens(x,cov,'centered_unit'),atol=0,rtol=0)
    assert not a[1].any() and not a[0,0].any()
    for mode in ARMS:
        assert not center_observed_tokens(torch.ones_like(x),torch.ones_like(cov),mode).any()
    with pytest.raises(ValueError): center_observed_tokens(x, cov+2, 'centered')


def test_same_parameters_initial_state_and_zero_bounded_prediction():
    for mode in ARMS:
        torch.manual_seed(17); a=CenteredTemporalDynamics(mode)
        torch.manual_seed(17); b=TemporalSourceDynamics('sequence')
        for key,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[key],atol=0,rtol=0)
        x=(torch.randn(2,480),torch.randn(2,8,512),torch.ones(2,8))
        assert not a(*x).any()
        with torch.no_grad(): a.head[-1].bias.fill_(1000)
        assert torch.all(torch.linalg.vector_norm(a(*x),dim=-1)<1)


@pytest.mark.parametrize('mode', ARMS)
def test_exact_resumed_training(mode,tmp_path):
    torch.manual_seed(4)
    x=(torch.randn(4,480),torch.randn(4,8,512),torch.ones(4,8))
    y=torch.randn(4,12,2)*.1; ids=np.arange(4); w=np.ones(4)/4
    def inputs(q): return (tuple(v[q] for v in x),(torch.ones(len(q)),torch.eye(2).expand(len(q),2,2),torch.ones(len(q),dtype=torch.bool)))
    config=dict(start_step=2,updates=6,batch_size=2,learning_rate=.0003,minimum_lr_ratio=.01,weight_decay=.0001,checkpoint_every=2)
    def run(path,stop=None):
        torch.manual_seed(17); model=CenteredTemporalDynamics(mode)
        result=fit_motion_candidate(model,inputs,lambda q:y[q],ids,w,scale=1.,seed=17,
            config=config,identity={'mode':mode},checkpoint=path,heartbeat=lambda **kw:None,
            suppress_zero_targets=False,stop_at=stop)
        return model,result
    full,_=run(tmp_path/'full.pt');run(tmp_path/'resumed.pt',2);resumed,result=run(tmp_path/'resumed.pt')
    for k,v in full.state_dict().items(): torch.testing.assert_close(v,resumed.state_dict()[k],atol=0,rtol=0)
    assert result['new_updates']==4 and run(tmp_path/'resumed.pt')[1]['new_updates']==0
