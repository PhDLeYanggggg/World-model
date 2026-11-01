import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_free_allocation_actor as api
from tests.test_m3w_coupled_benefit_risk import data as original_data


DECISION=dict(dual_every=2,record_dual_learning_rate=1.,query_dual_learning_rate=1.,dual_maximum=100.)


def settings():return dict(width=8,steps=6,query_batch_size=4,learning_rate=.001,gradient_clip=5.,heartbeat_every=2,checkpoint_every=2)


def fixture():
    source=original_data();old=api.prior.base.CoupledCritic(4,8,17,'released_benefit')
    state=dict(dimension=4,settings=dict(width=8),seed=17,mode='released_benefit',rms=source['rms'],model=copy.deepcopy(old.state_dict()))
    c=source['causal'];p=api.prior.base.predict(state,c)
    teacher=api.allocation.select_actions(p,c['allowed'],arm='physical_neural')[0]
    actor=api.causal_inputs(c,state,p,teacher,fitting=True)
    return api.training_data(actor,source),state,source


def test_unbounded_policy_can_represent_fallback_without_changing_moments():
    d,state,source=fixture();c=d['causal'];rows=np.arange(8);old=c['prediction'].copy()
    p=api.probabilities(torch.full((8,7),-80.,dtype=torch.float64),c,rows)
    assert (p[:,1:].sum(1)<1e-20).all()
    assert (p[:,0]>.999999999).all()
    np.testing.assert_array_equal(c['prediction'],old)


def test_all_seven_candidates_remain_available_not_only_teacher_choice():
    d,_,_=fixture();c=d['causal'];off=torch.zeros((8,7),dtype=torch.float64);off[:,6]=50
    p=api.probabilities(off,c,np.arange(8))
    assert (p[:,7]>.999999999).all() and (c['teacher']!=6).any()


def test_zero_actor_exact_native_utility_and_shared_initialization():
    d,_,_=fixture();c=d['causal'];m=api.AllocationActor(c['features']['self_context'].shape[-1],8,17)
    states=[]
    for arm in api.ARMS:
        states.append(dict(arm=arm,dimension=c['features'][arm].shape[-1],settings=settings(),seed=17,model=m.state_dict(),preprocess=c['preprocess'],temperature=c['temperature']))
    for state in states:np.testing.assert_array_equal(api.predict(state,c),c['native_gain'])
    assert not np.array_equal(c['features']['self_context'],c['features']['selected_context'])


def test_inference_requires_frozen_preprocessing_and_rejects_future_fields(tmp_path):
    d,state,source=fixture();c=d['causal']
    with pytest.raises(ValueError):api.causal_inputs(source['causal'],state,c['prediction'],c['teacher'])
    same=api.causal_inputs(source['causal'],state,c['prediction'],c['teacher'],preprocess=c['preprocess'])
    for arm in api.ARMS:np.testing.assert_array_equal(c['features'][arm],same['features'][arm])
    fitted=api.fit(d,arm='selected_context',seed=17,settings=settings(),decision=DECISION,identity={},path=tmp_path/'x.pt.gz')
    for key in ('future_endpoint','known','targets'):
        with pytest.raises(ValueError):api.predict(fitted,dict(c,**{key:np.zeros(8)}))
    with pytest.raises(AssertionError):api.predict(fitted,dict(c,temperature=2*c['temperature']))


def test_full_query_and_record_pass_matches_explicit_sums():
    d,_,_=fixture();c=d['causal'];m=api.AllocationActor(c['features']['self_context'].shape[-1],8,17)
    with torch.no_grad():p=api.probabilities(m(torch.from_numpy(c['features']['self_context'])),c,np.arange(8))[:,1:].numpy()
    r=np.zeros((len(d['layout']['record_keys']),3));q=np.zeros((len(c['groups']),2))
    np.add.at(r,d['layout']['record'],(p[:,:,None]*d['layout']['risk']).sum(1));r[:,2]-=d['layout']['easy_constant']
    np.add.at(q,d['query_index'],(p[:,:,None]*d['predicted_query_risk']).sum(1))
    rr,qq,mass=api.full_risk(m,d,'self_context',chunk_rows=3)
    np.testing.assert_allclose(rr,r/d['layout']['record_scales'][:,None],rtol=1e-14)
    np.testing.assert_allclose(qq,q/d['query_scales'][:,None],rtol=1e-14)
    assert mass==pytest.approx(p.sum()/8)


@pytest.mark.parametrize('arm',api.ARMS)
def test_exact_resume_unknown_labels_never_inputs_and_cost_model_is_frozen(tmp_path,arm):
    d,old,source=fixture();unchanged=copy.deepcopy(old);kw=dict(arm=arm,seed=17,settings=settings(),decision=DECISION,identity={})
    direct=api.fit(d,path=tmp_path/'direct.pt.gz',**kw)
    api.fit(d,path=tmp_path/'resume.pt.gz',stop_at=2,**kw)
    resumed=api.fit(d,path=tmp_path/'resume.pt.gz',resume=True,**kw)
    for key in direct:
        if key!='seconds':api.parent.exact(direct[key],resumed[key])
    api.parent.exact(unchanged,old)
    assert direct['unknown_bound_rows_drawn']>0 and direct['unknown_rows_with_supervised_labels']==0
    assert any(not torch.equal(v,direct['initial_model'][k]) for k,v in direct['model'].items())


def test_context_arms_have_identical_initialization_draws_and_budget(tmp_path):
    d,_,_=fixture();states=[api.fit(d,arm=a,seed=17,settings=settings(),decision=DECISION,identity={},path=tmp_path/(a+'.pt.gz')) for a in api.ARMS]
    api.assert_matched(*states);assert states[0]['parameters']==185


def test_shared_preprocessing_is_fitted_without_outcome_mask():
    d,old,source=fixture();source2=copy.deepcopy(source);source2['targets'][:]=np.nan
    c=api.causal_inputs(source2['causal'],old,d['causal']['prediction'],d['causal']['teacher'],fitting=True)
    api.parent.exact(c['preprocess'],d['causal']['preprocess'])
    assert not c['preprocess']['known_outcome_mask_used']
