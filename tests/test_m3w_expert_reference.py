import inspect
import numpy as np
import pytest
import torch
from src.world_model import m3w_expert_reference as m
from tests.test_m3w_role_separated_cost import data as raw_data


def prepared():
    d=raw_data();rng=np.random.default_rng(3);d['context']=rng.normal(size=(len(d['ids']),70))
    p=m.parent.prepare(d,'a');torch.manual_seed(17);net=m.parent.Head(m.parent.SETTINGS,p['binding'])
    return d,dict(model=net.state_dict(),preprocess=p['binding'],arm='role_separated',complete=True,step=4096)


def test_anchor_join_is_order_independent_and_first50_stay_shared():
    ids=np.array([15,13,8,20]);x=np.zeros((4,120),np.float32)
    x[:,50:]=np.arange(4)[:,None]
    table=m.context_table(x,ids)
    np.testing.assert_array_equal(m.lookup(table,ids)[:,0],[3,1,1,3])
    with pytest.raises(ValueError):m.context_table(x[:1],ids[:1])
    x[0,0]=1
    with pytest.raises(AssertionError):m.context_table(x,ids)


def test_both_arms_start_at_old_reference_with_zero_new_context_effect():
    d,o=prepared();p=m.prepare(d,'a',o);model=m.Reference(o)
    a=model(p['x'][:,:50],p['context'],'history_refit')
    b=model(p['x'][:,:50],p['context'],'expert_context')
    assert torch.equal(a,b) and model.context.weight.count_nonzero()==0
    old=m.parent.Head(m.parent.SETTINGS,o['preprocess']);old.load_state_dict(o['model'])
    torch.testing.assert_close(a,old(p['x'],p['envelope'])['conditional'][:,2:4],rtol=0,atol=0)


def test_held_source_has_no_effect_on_any_learned_preprocessing():
    d,o=prepared();a=m.prepare(d,'a',o);d['context'][:30]=1e5
    d['moments'][:30]=1e6;d['observed'][:30]=~d['observed'][:30];d['x'][:30]=1e5
    b=m.prepare(d,'a',o);m.base.exact(a,b)


def test_model_never_reads_candidate_dependent_fields_or_future_input():
    d,o=prepared();p=m.prepare(d,'a',o);model=m.Reference(o)
    with torch.no_grad():
        model.context.weight.normal_();model.output.weight.normal_()
    x=p['x'][:3].clone().requires_grad_();c=p['context'][:3]
    out=model(x[:,:50],c,'expert_context');grad=torch.autograd.grad(out.sum(),x)[0]
    assert grad[:,50:].count_nonzero()==0
    assert not torch.equal(out,model(x[:,:50],c+1,'expert_context'))
    assert torch.equal(model(x[:,:50],c,'history_refit'),model(x[:,:50],c+1,'history_refit'))
    assert not {'target','valid','observed','future_endpoint'}&set(inspect.signature(m.predict).parameters)


def test_frozen_harm_and_support_survive_reference_replacement():
    p=np.ones((2,7,5));p[...,0]=.1;p[...,3]=.5;p[...,4]=.25;q=np.full((2,7),.5)
    bank=dict(conditional=p,moments=p*.5,probability=q);ok=np.ones((2,7),bool);ok[:,0]=False
    r=np.ones((2,7,2))*.02;r[...,1]=.01
    z=m.replace_reference(bank,r,ok)
    for j in (1,4):np.testing.assert_array_equal(z['moments'][...,j],bank['moments'][...,j])
    np.testing.assert_array_equal(z['probability'],q)
    assert (z['conditional'][...,0][ok]<=r[...,0][ok]).all()
    assert (z['conditional'][...,3][ok]<=z['conditional'][...,2][ok]).all()


def test_real_optimization_recovers_exactly_and_arms_share_draws(tmp_path):
    d,o=prepared()
    m.fit(d,'a',o,{},tmp_path/'r.gz',arm='expert_context',seed=29,stop_at=2)
    a=m.fit(d,'a',o,{},tmp_path/'r.gz',arm='expert_context',seed=29,stop_at=4,resume=True)
    b=m.fit(d,'a',o,{},tmp_path/'d.gz',arm='expert_context',seed=29,stop_at=4)
    c=m.fit(d,'a',o,{},tmp_path/'c.gz',arm='history_refit',seed=29,stop_at=4)
    m.exact(a,b)
    for k in ('preprocess','initial_model','generator','visits','step','parameters'):m.base.exact(a[k],c[k])
    assert a['model']['context.weight'].count_nonzero()>0 and c['model']['context.weight'].count_nonzero()==0
