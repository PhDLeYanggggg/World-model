import copy
import numpy as np
import pytest
from src.world_model import m3w_reference_cost_refit as api
from tests.test_m3w_coupled_benefit_risk import data


def settings():return dict(width=8,largest_source_cycles=2,minimum_updates=4,query_batch_size=4,learning_rate=.001,gradient_clip=5.,heartbeat_every=1,checkpoint_every=1)


def warm(d):
    m=api.base.CoupledCritic(4,8,17,'released_benefit')
    return dict(dimension=4,settings=dict(width=8),seed=17,model=m.state_dict(),input_hashes=d['input_hashes'])


def test_cycle_covers_all_queries_and_resume_draws_exact():
    groups=[np.array([i]) for i in range(11)];sources=[np.arange(9),np.arange(9,11)]
    direct=api.CyclicQueries(groups,sources,4,17);counts=[]
    for _ in range(5):counts.append(direct.draw()[2])
    state=direct.state();resume=api.CyclicQueries(groups,sources,4,17);resume.restore(state)
    for _ in range(15):
        a,b=direct.draw(),resume.draw()
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert direct.counts.min()>=4
    api.parent.exact(direct.state(),resume.state())


def test_initial_forward_and_native_predict_reproduce_warm_costs_exact():
    d=data();w=warm(d);args=api.parent.batch(d['causal'],np.arange(8),np.repeat(np.arange(4),2),4)
    old=api.base.CoupledCritic(4,8,17,'released_benefit');old.load_state_dict(w['model'])
    states=[]
    for arm in api.ARMS:
        m=api.initialize(w,arm);api.torch.testing.assert_close(m(*args),old(*args),rtol=0,atol=0)
        state=dict(dimension=4,settings=settings(),seed=17,arm=arm,model=m.state_dict());states.append(state)
        np.testing.assert_array_equal(api.predict(state,d['causal']),d['causal']['prediction'])
    api.parent.exact(states[0]['model'],states[1]['model'])


def test_only_unlocked_branch_can_change_references_and_cost_bounds_remain():
    d=data();args=api.parent.batch(d['causal'],np.arange(8),np.repeat(np.arange(4),2),4)
    for arm in api.ARMS:
        m=api.initialize(warm(d),arm)
        with api.torch.no_grad():m.reference_output.bias.copy_(api.torch.tensor([.4,-.4]))
        p=m(*args)
        assert api.torch.isfinite(p).all() and (p>=0).all() and (p[:,:,3]<=p[:,:,2]).all()
        assert (p[:,:,4]<=p[:,:,1]).all() and (p[:,:,:2].sum(2)<=args[2]+1e-12).all()
        assert api.torch.equal(p[:,:,[2,3]],args[1][:,:,[2,3]])==(arm=='locked_references')


@pytest.mark.parametrize('arm',api.ARMS)
def test_actual_optimizer_updates_and_interruption_recovery_exact(tmp_path,arm):
    d=data();w=warm(d);kw=dict(data=d,warm=w,arm=arm,settings=settings(),identity={})
    direct=api.fit(path=tmp_path/'direct.pt.gz',**kw)
    api.fit(path=tmp_path/'resume.pt.gz',stop_at=1,**kw)
    recovered=api.fit(path=tmp_path/'resume.pt.gz',resume=True,**kw)
    for key in direct:
        if key!='seconds':api.parent.exact(direct[key],recovered[key])
    assert direct['sampler']['counts'].min()>=2 and direct['unknown_rows_with_supervised_labels']==0
    assert direct['unknown_rows_drawn']>0
    assert any(not api.torch.equal(v,direct['initial_model'][k]) for k,v in direct['model'].items())
    with pytest.raises(ValueError):api.predict(direct,dict(d['causal'],future_endpoint=np.zeros((8,2))))


def test_paired_arms_match_initialization_sampling_and_budget(tmp_path):
    d=data();w=warm(d);states=[api.fit(d,warm=w,arm=a,settings=settings(),identity={},path=tmp_path/(a+'.pt.gz')) for a in api.ARMS]
    api.assert_matched(*states)
    assert states[1]['trainable_parameters']-states[0]['trainable_parameters']==2*(2*8+2)+2
    assert states[0]['total_parameters']==states[1]['total_parameters']


def test_budget_never_reduces_registered_minimum_updates():
    d=data();s=settings();s['minimum_updates']=2000
    assert api.steps_for(d['causal'],s)==2000


def test_causal_schema_and_sampling_partition_cannot_silently_change(tmp_path):
    d=data();w=warm(d);d['input_hashes']=copy.deepcopy(d['input_hashes']);d['input_hashes']['targets']='changed'
    with pytest.raises((ValueError,AssertionError)):api.fit(d,warm=w,arm=api.ARMS[0],settings=settings(),identity={},path=tmp_path/'bad.pt.gz')
    with pytest.raises(ValueError):api.CyclicQueries([np.array([0])],[np.array([0]),np.array([0])],4,17)
