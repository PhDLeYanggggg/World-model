import inspect
import numpy as np
import pytest
from src.evaluation import m3w_fixed_amplitude_controls as m


def packet():
    h=np.zeros((3,8,2));h[1,:,0]=np.arange(8);h[2,-2,0]=1
    b=np.zeros((3,12,2));p=np.stack([b,*[b+a for a in m.FRACTIONS]],axis=1)
    p[2]=0
    return h,p,b


def test_stationary_guard_uses_last_past_delta_only_not_eventual_labels():
    h,p,b=packet()
    assert m.choices(h,p,b,1,'all_past_eligible').tolist()==[1,1,-1]
    assert m.choices(h,p,b,1,'last_velocity_nonzero').tolist()==[-1,1,-1]
    assert not {'target','valid','future','endpoint'}&set(inspect.signature(m.choices).parameters)
    h[:,:-2]=1e6
    assert m.choices(h,p,b,1,'last_velocity_nonzero').tolist()==[-1,1,-1]


def test_no_new_tiny_actions_and_no_fake_zero_intervention():
    h,p,b=packet()
    assert len(m.variants())==12
    with pytest.raises(ValueError):m.choices(h,p,b,0,'all_past_eligible')
    with pytest.raises(ValueError):m.choices(h,p,b,7,'all_past_eligible')
    with pytest.raises(ValueError):m.choices(h[:,:7],p,b,1,'all_past_eligible')
    p[:]=0
    assert (m.choices(h,p,b,1,'all_past_eligible')==-1).all()


def test_appended_future_cannot_be_silently_used_as_history():
    h,p,b=packet()
    with pytest.raises(ValueError):m.choices(np.concatenate((h,h),axis=1),p,b,1,'last_velocity_nonzero')


def test_static_uncertainty_does_not_invent_seed_replication_or_drop_na():
    rows=[dict(source=str(i),gain=1.) for i in range(8)]
    a=m.interval(rows,'gain',[str(i) for i in range(8)])
    assert a['point']==1 and not a['risk_head_seeds_averaged']
    with pytest.raises(ValueError):m.interval(rows*3,'gain',[str(i) for i in range(8)])
    rows[0]['gain']=None
    assert m.interval(rows,'gain',[str(i) for i in range(8)])['point'] is None


def test_no_intervention_does_not_receive_identity_exception():
    from src.evaluation.m3w_selection_forecast_readout import policy
    floor=np.zeros((1,12,2));target=np.ones_like(floor);valid=np.ones((1,12),bool)
    r=policy(floor,floor,floor,target,valid,np.array(['r']),easy_cut=2,hard_cut=3)['recordings']['r']
    assert r['known_easy_net_degradation']==0 and r['known_easy_risk'] is None and not r['supported']
