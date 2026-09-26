import inspect
import numpy as np
import pytest
import torch
from src.world_model import m3w_membership_auxiliary as m
from src.world_model import m3w_selected_risk_learning as old
from tests.test_m3w_membership_cost import fixture


@pytest.mark.parametrize('arm', ['cost_only','membership_aux'])
def test_exact_resume_and_unknown(tmp_path, arm):
    torch.set_num_threads(1); x,y,e,s,d,p,c=fixture()
    kw=dict(arm=arm,seed=17,settings=c,identity={'test':1},heartbeat=lambda **_:None)
    a,r=m.fit(x,y,e,s,d,p,directory=tmp_path/'a',**kw)
    m.fit(x,y,e,s,d,p,directory=tmp_path/'b',stop_at=4,**kw)
    b,_=m.fit(x,y,e,s,d,p,directory=tmp_path/'b',resume=True,**kw)
    for u,v in zip(m.predict(a,x,d,p),m.predict(b,x,d,p)): np.testing.assert_array_equal(u,v)
    assert r['unknown_rows_sampled']==0
    with pytest.raises(ValueError): m.fit(x,y,e,s,d,p,directory=tmp_path/'a',resume=True,**dict(kw,identity={}))


def test_cost_control_matches_original(tmp_path):
    torch.set_num_threads(1); x,y,e,s,d,p,c=fixture(); masks=np.zeros((len(x),3),bool)
    kw=dict(seed=29,settings=c,identity={},heartbeat=lambda **_:None)
    a,_=m.fit(x,y,e,s,d,p,arm='cost_only',directory=tmp_path/'a',**kw)
    b,_=old.fit(x,y,s,d,masks,p,arm='mean',directory=tmp_path/'b',**kw)
    np.testing.assert_allclose(m.predict(a,x,d,p)[0],old.predict(b,x,d,p),rtol=1e-6,atol=1e-7)
    u=m.restore(tmp_path/'a')[1]; v=torch.load(tmp_path/'b/checkpoint.pt',weights_only=False)
    for k in ('draws','fixed_ids','loss_scales'): np.testing.assert_array_equal(u[k],v[k])
    assert torch.equal(u['sampler_rng'],v['sampler_rng'])


def test_auxiliary_does_not_multiply_costs():
    assert 'easy' not in inspect.signature(m.predict).parameters
    a=m.AuxiliaryCostHead(3,4); x=torch.randn(5,3); d=torch.ones(5)
    before,p=a(x,d)
    with torch.no_grad(): a.membership.weight.add_(100); a.membership.bias.add_(100)
    after,q=a(x,d)
    assert torch.equal(before,after) and not torch.equal(p,q)
    assert torch.isfinite(after).all()


def test_auxiliary_changes_shared_encoder(tmp_path):
    x,y,e,s,d,p,c=fixture(); outs=[]
    for arm in ('cost_only','membership_aux'):
        model,_=m.fit(x,y,e,s,d,p,arm=arm,seed=17,settings=c,identity={},directory=tmp_path/arm,heartbeat=lambda **_:None)
        outs.append(model.network[0].weight.detach())
    assert not torch.equal(*outs)


def test_invalid_labels_fail(tmp_path):
    x,y,e,s,d,p,c=fixture(); e[0]=.2
    with pytest.raises(ValueError): m.fit(x,y,e,s,d,p,arm='membership_aux',seed=1,settings=c,identity={},directory=tmp_path,heartbeat=lambda **_:None)
