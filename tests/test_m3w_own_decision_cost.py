import copy
import numpy as np
import pytest
from src.world_model import m3w_own_decision_cost as api
from src.world_model.m3w_coherent_cost_projection import project
from tests.test_m3w_reference_cost_refit import data,warm as original_warm,settings as original_settings


def settings():return dict(original_settings(),solver_seconds_per_query=2.)


def warm(d):
    w=original_warm(d);m=api.previous.initialize(w,'learned_references')
    return dict(w,arm='learned_references',model=m.state_dict())


def test_native_coherence_matches_numpy_semantics_and_preserves_harm():
    rng=np.random.default_rng(3);p=rng.uniform(.01,5,(20,7,5));p[:,:,4]=np.minimum(p[:,:,4],p[:,:,1]);ok=rng.random((20,7))>.3;ok[0]=False
    t=api.torch.tensor(p,requires_grad=True);actual=api.coherent_native(t,api.torch.tensor(ok))
    np.testing.assert_allclose(actual.detach(),project(p,ok),rtol=0,atol=1e-14)
    np.testing.assert_array_equal(actual.detach()[:,:,[1,4]],p[:,:,[1,4]])
    np.testing.assert_array_equal(actual.detach()[~ok],p[~ok]);actual.sum().backward()
    assert api.torch.isfinite(t.grad).all()
    assert (actual[:,:,0][ok]<=actual[:,:,2][ok]).all()


def test_both_arms_share_the_same_warm_coherent_native_predictions():
    d=data();w=warm(d);model=api.initialize(w);args,anchor,scale,allowed=api.batch(d['causal'],np.arange(8),np.repeat(np.arange(4),2),4)
    value=model.native(args,anchor,scale,allowed).detach().numpy()
    np.testing.assert_allclose(value,project(api.previous.predict(w,d['causal']),d['causal']['allowed']),rtol=0,atol=1e-12)
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert (value[:,:,:2].sum(2)<=d['causal']['envelope']+1e-10).all()


def test_fixed_query_arithmetic_reproduces_existing_loss_and_unknown_gradient_zero():
    d=data();rows=np.arange(8);segments=api.torch.repeat_interleave(api.torch.arange(4),2);c=d['causal']
    p=api.torch.tensor(c['prediction']/c['scales'][:,None,None],requires_grad=True)
    norm=api.previous.base.units.query_normalizer(d)
    old=api.previous.losses(p,d,rows,segments,4,norm)
    new=api.losses(p,d,rows,segments,4,norm,c['initial'])
    api.torch.testing.assert_close(new['total'],old['total'],rtol=0,atol=0)
    new['total'].backward();assert (p.grad[~d['known']]==0).all()
    assert new['unknown_supervised_rows']==0


def test_own_allocation_is_causal_and_uses_original_exact_count_risk_contract():
    d=data();p=project(d['causal']['prediction'],d['causal']['allowed']);seg=np.repeat(np.arange(4),2)
    c,info=api.own_choices(p,d['causal']['allowed'],seg,seconds=2.)
    independent,_=api.allocation.base.select_actions(p,d['causal']['allowed'],arm='physical_neural')
    assert info['queries']==4 and not info['labels_used']
    for k in range(4):
        at=seg==k;assert api.allocation.base.feasible(p[at],d['causal']['allowed'][at],c[at])
        assert (c[at]>=0).sum()==(independent[at]>=0).sum()


@pytest.mark.parametrize('arm',api.ARMS)
def test_real_optimizer_and_exact_interruption_recovery(tmp_path,arm):
    d=data();kw=dict(data=d,warm=warm(d),arm=arm,settings=settings(),identity={})
    direct=api.fit(path=tmp_path/'direct.pt.gz',**kw)
    api.fit(path=tmp_path/'resume.pt.gz',stop_at=1,**kw)
    resumed=api.fit(path=tmp_path/'resume.pt.gz',resume=True,**kw)
    api.assert_replay(direct,resumed)
    assert direct['step']==direct['budget'] and direct['sampler']['counts'].min()>=2
    assert direct['unknown_rows_with_supervised_labels']==0
    assert any(not api.torch.equal(v,direct['initial_model'][k]) for k,v in direct['model'].items())
    with pytest.raises(ValueError):api.predict(direct,dict(d['causal'],future_endpoint=np.zeros((8,2))))


def test_matched_active_capacity_draws_and_budget(tmp_path):
    d=data();states=[api.fit(d,warm=warm(d),arm=a,settings=settings(),identity={},path=tmp_path/(a+'.pt.gz')) for a in api.ARMS]
    api.assert_matched(*states)
    assert states[0]['training_solver_calls']==0 and states[1]['training_solver_calls']==16
    assert states[0]['total_parameters']==states[0]['trainable_parameters']==states[1]['trainable_parameters']


def test_warm_lineage_and_causal_schema_cannot_change(tmp_path):
    d=data();w=warm(d);w['input_hashes']=copy.deepcopy(w['input_hashes']);w['input_hashes']['targets']='changed'
    with pytest.raises((ValueError,AssertionError)):api.fit(d,warm=w,arm=api.ARMS[0],settings=settings(),identity={},path=tmp_path/'bad.pt.gz')
    with pytest.raises(ValueError):api.initialize(dict(w,arm='locked_references'))


def test_replay_excludes_only_measured_wall_time_not_numerical_state():
    a=dict(model=api.torch.ones(2),trace=[dict(loss=1.,training_solver_seconds=2.)],seconds=3.)
    b=dict(model=api.torch.ones(2),trace=[dict(loss=1.,training_solver_seconds=5.)],seconds=6.)
    api.assert_replay(a,b);b['trace'][0]['loss']=1.01
    with pytest.raises(AssertionError):api.assert_replay(a,b)
