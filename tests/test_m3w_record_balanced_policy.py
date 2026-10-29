import pytest

from src.world_model import m3w_record_balanced_policy as api
from tests.test_m3w_direct_policy import fixture, settings

torch=api.torch


def test_constraint_weights_uniform_gain_weights_unchanged():
    w=torch.tensor([.1,.2,.7]);initial=w.clone()
    factor=api.constraint_factor(w)
    torch.testing.assert_close(w*factor,torch.full_like(w,1./3))
    assert torch.equal(w,initial)
    with pytest.raises(ValueError):api.constraint_factor(torch.tensor([0.,1.]))
    with pytest.raises(ValueError):api.constraint_factor(torch.tensor([.2,.2]))


def test_chunked_loss_and_gradient_match_uniform_record_penalty():
    data=api.base.prepare(*fixture())
    w=torch.arange(1,len(data['groups'])+1,dtype=torch.float32)
    data['weights']=w/w.sum()
    loc=api.previous.whole.layout(data)
    m=api.base.Policy(data['z'].shape[1],8,17);dual=torch.full((len(data['groups']),2),.7)
    actual=api.full_pass(m,data,loc,dual,.5,2,True)
    g=torch.cat([p.grad.flatten() for p in m.parameters()]);m.zero_grad(set_to_none=True)
    ix,rec=loc['index'],loc['record'];z=m(data['z'][ix],data['anchor'][ix])
    risk=(dual[rec]*data['risk'][ix]/data['record_scales'][rec]).sum(1)
    n=len(data['groups']);row_weight=loc['weight']
    coefficient=-data['gain'][ix]+risk/(n*data['weights'][rec])
    ref=(api.previous.cost_surrogate(z,coefficient,.5)*row_weight).sum();ref.backward()
    expected=torch.cat([p.grad.flatten() for p in m.parameters()])
    torch.testing.assert_close(g,expected,atol=1e-7,rtol=1e-6)
    assert actual['surrogate_objective']==pytest.approx(float(ref.detach()))
    hard=(z>0).double();risk_term=(hard*risk/loc['sizes'][rec]).sum()/n
    utility_term=-(hard*data['gain'][ix]*row_weight).sum()
    assert actual['hard_objective']==pytest.approx(float(utility_term+risk_term),abs=1e-7)


def test_neutral_engineering_replay_matches_parent_training(tmp_path):
    d=api.base.prepare(*fixture());kw=dict(settings={**settings(),'chunk_size':2},seed=17,identity={'test':True})
    original=api.previous.fit(d,path=tmp_path/'old.pt.gz',**kw)
    neutral=api.fit(d,path=tmp_path/'neutral.pt.gz',balanced=False,**kw)
    for key in ('model','optimizer','dual','torch_rng','trace','initial_model','input_hashes'):
        api.base.core.exact(original[key],neutral[key])


def test_resume_preserves_all_training_state(tmp_path):
    d=api.base.prepare(*fixture());kw=dict(settings={**settings(),'chunk_size':2},seed=17,identity={'test':True})
    complete=api.fit(d,path=tmp_path/'direct.pt.gz',**kw)
    api.fit(d,path=tmp_path/'resume.pt.gz',stop_at=2,**kw)
    resumed=api.fit(d,path=tmp_path/'resume.pt.gz',resume=True,**kw)
    for key in complete:
        if key!='seconds':api.base.core.exact(complete[key],resumed[key])
    assert any(not torch.equal(v,complete['initial_model'][k]) for k,v in complete['model'].items())
