import copy
import numpy as np
import pytest
import torch
from src.evaluation import m3w_task_gradients as m
from src.world_model import m3w_membership_auxiliary as aux
from tests.test_m3w_membership_cost import fixture


def trained(tmp_path, arm='membership_aux'):
    torch.set_num_threads(1)
    x,y,e,s,d,p,c = fixture()
    aux.fit(x,y,e,s,d,p,arm=arm,seed=17,settings=c,identity={},directory=tmp_path,heartbeat=lambda **_:None)
    model,state = aux.restore(tmp_path)
    return model,state,m.fitting_batch(x,d,y,e,s,'held',state),(x,y,e,s,d)


def equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for v,w in zip(a,b): equal(v,w)
    else: assert a==b


@pytest.mark.parametrize('arm',['cost_only','membership_aux'])
def test_diagnostic_is_repeatable_non_mutating(tmp_path,arm):
    model,state,batch,_=trained(tmp_path,arm)
    original=copy.deepcopy(state); weights=copy.deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
    first=m.diagnose(model,state,batch); second=m.diagnose(model,state,batch)
    assert first==second and first['gradient_additivity_checked']
    equal(state,original); equal(weights,model.state_dict()); assert torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    if arm=='cost_only':
        assert first['shared_BCE_relation']['cost']['norm']==0
        assert first['shared_BCE_relation']['cost']['cosine'] is None


def test_batch_rejects_held_and_unknown(tmp_path):
    _,state,_,(x,y,e,s,d)=trained(tmp_path)
    with pytest.raises(ValueError): m.fitting_batch(x,d,y,e,s,'a',state)
    state['fixed_ids'][0]=7
    with pytest.raises(ValueError): m.fitting_batch(x,d,y,e,s,'held',state)


def test_virtual_step_matches_manual_adam(tmp_path):
    model,state,batch,_=trained(tmp_path); report,delta=m.virtual_step(model,state,batch,'membership_aux')
    reference=copy.deepcopy(model); opt=torch.optim.AdamW(reference.parameters())
    opt.load_state_dict(copy.deepcopy(state['optimizer'])); opt.zero_grad(set_to_none=True)
    z,d,y,e,s=batch; pred,logit=reference(z,d)
    loss,_=aux.objective(pred,logit,y,e,s,'membership_aux'); loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(),state['settings']['gradient_clip']); opt.step()
    expected=torch.cat([(q-p).detach().reshape(-1).double() for p,q in zip(model.parameters(),reference.parameters())]).numpy()
    np.testing.assert_array_equal(delta,expected); assert report['after']==m.values(reference,batch)


def test_component_gradients_match_finite_difference(tmp_path):
    model,state,batch,_=trained(tmp_path)
    model.double(); batch=tuple(t.double() for t in batch)
    cost,_,_=m.terms(model,batch); p=model.network[0].weight
    grad=torch.autograd.grad(cost,p)[0]; direction=torch.ones_like(p)/p.numel()**.5
    eps=1e-5
    with torch.no_grad():
        old=p.clone(); p.copy_(old+eps*direction); plus=m.values(model,batch)['cost']
        p.copy_(old-eps*direction); minus=m.values(model,batch)['cost']; p.copy_(old)
    assert (plus-minus)/(2*eps)==pytest.approx(float((grad*direction).sum()),rel=1e-5,abs=1e-8)


def test_population_preserves_undefined_and_effect_tolerance(tmp_path):
    model,state,batch,_=trained(tmp_path,'cost_only'); row=m.diagnose(model,state,batch)
    p=m.population([row]); assert p['gradients']['cost']['cosine']['defined']==0
    for k in ('cost','H','H_E'):
        assert sum(p['optimizer'][k][v] for v in ('auxiliary_worse','auxiliary_better','tied'))==1
