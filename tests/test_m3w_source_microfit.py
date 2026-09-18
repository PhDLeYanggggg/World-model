import numpy as np
import pytest
import torch

from src.world_model.m3w_source_cost_dynamics import SourceDynamics, restore
from src.world_model.m3w_source_microfit import MicrofitDynamics, fit_micro, metrics, select_cohorts


def payload():
    torch.manual_seed(2)
    features=(torch.randn(4,6),torch.rand(4,8,3,32,32),torch.ones(4,8,1,32,32))
    frame=(torch.tensor([10.,20.,100.,0.]),torch.eye(2)[None].repeat(4,1,1),torch.tensor([True,True,True,False]))
    target=torch.randn(4,12,2);target[-1]=0
    return features,frame,target


def test_original_decoder_is_exact_and_both_initialize_at_cv():
    features,frame,_=payload();torch.manual_seed(17);model=MicrofitDynamics(6)
    torch.manual_seed(17);old=SourceDynamics(6)
    for p,q in zip(model.parameters(),old.parameters()):assert torch.equal(p,q)
    for decoder in ('context_radius','training_cost_scale'):
        assert torch.count_nonzero(model.trajectory(features,frame,decoder,2.))==0
    with torch.no_grad():
        model.head[-1].bias.fill_(.2);old.load_state_dict(model.state_dict())
    assert torch.equal(model.trajectory(features,frame,'context_radius',2.),restore(old(*features,'past_rgb'),*frame))


def test_rescaling_preserves_context_bound_and_missing_context():
    features,frame,_=payload();model=MicrofitDynamics(6)
    with torch.no_grad():model.head[-1].bias.fill_(100.)
    for decoder in ('context_radius','training_cost_scale'):
        p=model.trajectory(features,frame,decoder,2.)
        assert torch.all(torch.linalg.vector_norm(p[:3],dim=-1)<frame[0][:3,None])
        assert torch.count_nonzero(p[3])==0
        p.sum().backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        model.zero_grad()


def test_origin_output_jacobian_changes_not_evaluation_coordinate():
    features,frame,_=payload();model=MicrofitDynamics(6)
    grads=[]
    for decoder in ('context_radius','training_cost_scale'):
        p=model.trajectory(features,frame,decoder,2.)
        grads.append(torch.autograd.grad(p[0,0,0],model.head[-1].bias)[0][0].item())
    assert grads==pytest.approx([10.,2.])


def test_fullbatch_resume_is_exact_and_completed_resume_is_readonly(tmp_path):
    torch.set_num_threads(2);features,frame,target=payload()
    config=dict(updates=7,learning_rate=.0003,weight_decay=.0001,checkpoint_every=2)
    def run(name,stop=None):
        torch.manual_seed(17);model=MicrofitDynamics(6)
        return fit_micro(model,features,frame,target,decoder='training_cost_scale',scale=2.,
            config=config,identity={'split':'training'},checkpoint=tmp_path/name,heartbeat=lambda **_:None,stop_at=stop)
    a,_=run('full.pt');run('resume.pt',3);b,r=run('resume.pt')
    np.testing.assert_array_equal(a,b);assert r['new_updates']==4
    before=(tmp_path/'resume.pt').read_bytes();assert run('resume.pt')[1]['new_updates']==0
    assert before==(tmp_path/'resume.pt').read_bytes()


def test_easy_ratio_remains_undefined_and_no_target_input_argument():
    features,frame,target=payload();m=MicrofitDynamics(6)
    with pytest.raises(TypeError):m.trajectory(features,frame,'context_radius',2.,future_endpoint=target)
    v=metrics(torch.zeros_like(target),target,2.)
    assert v['gain_percent']==0 and v['easy_relative_degradation'] is None


def test_microfit_cohorts_share_positive_tracks_and_exclude_unsupported_or_infeasible():
    ids=np.arange(20)+100;tracks=np.array([f't{i}' for i in range(20)])
    cv=np.array([1.]*10+[0.]*10);extent=cv.copy();radius=np.full(20,10.);support=np.ones(20,bool)
    support[0]=False;extent[1]=9.
    sets=select_cohorts(ids,tracks,cv,extent,radius,support,count=4)
    assert len(sets['nonzero_only'])==4 and len(sets['mixed_zero'])==8
    np.testing.assert_array_equal(sets['mixed_zero'][:4],sets['nonzero_only'])
    assert 100 not in sets['mixed_zero'] and 101 not in sets['mixed_zero']
    assert np.all(cv[sets['nonzero_only']-100]>0)
    assert np.all(cv[sets['mixed_zero'][4:]-100]==0)
    with pytest.raises(ValueError):select_cohorts(ids,tracks,cv,extent,radius,support,count=11)
