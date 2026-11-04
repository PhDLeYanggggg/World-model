import numpy as np
import pytest

from src.world_model import m3w_harm_refinement as api
from tests.test_m3w_amplitude_cost import packet


def fixture():
    inputs,target,valid,pr=packet()
    bank=api.parent.causal_bank(**inputs)
    data=api.parent.training_data(bank,target,valid,1.,pr)
    p=api.parent.targets(bank,target,valid,1.)['moments'].copy()
    p[:,:,1]*=.01;p[:,:,4]*=.01
    return data,p


def test_upward_projection_keeps_references_and_cannot_create_feasible_actions():
    data,p=fixture();c=data['causal'];d=np.ones((*p.shape[:2],2))*7
    new=api.update(p,d,c['envelope'],c['allowed'])
    np.testing.assert_array_equal(p[:,:,[2,3]],new[:,:,[2,3]])
    assert (new[:,:,1]>=p[:,:,1]-1e-10).all()and (new[:,:,4]>=p[:,:,4]-1e-10).all()
    assert (new[:,:,0]<=p[:,:,0]+1e-10).all()
    assert (new[:,:,4]<=new[:,:,1]+1e-10).all()
    assert (new[:,:,0]+new[:,:,1]<=c['envelope']+1e-6).all()
    feasible=lambda a:(a[:,:,0]>a[:,:,1])&(a[:,:,2]>0)&(a[:,:,3]>0)&(a[:,:,1]<=.02*a[:,:,2])&(a[:,:,4]<=.02*a[:,:,3])&c['allowed']
    assert not (feasible(new)&~feasible(p)).any()
    np.testing.assert_allclose(api.update(p,-d,c['envelope'],c['allowed']),p,rtol=0,atol=1e-13)


def test_region_weights_preserve_source_and_query_mass():
    data,p=fixture();c=data['causal'];rows,actions,q=api.layout(c)
    choice=np.full(len(p),6)
    for arm in api.ARMS:
        w=api.weights(c,choice,arm,4.)
        for source in c['source_groups']:
            for query in source:
                if np.any(q[rows]==query):
                    assert w[q[rows]==query].sum()==pytest.approx(1/len(c['source_groups'])/len(source))


def test_all_arms_exact_resume_and_no_future_inputs(tmp_path):
    data,p=fixture();c=data['causal']
    settings=dict(rounds=4,learning_rate=.5,max_depth=3,max_features=64,min_samples_leaf=1,region_multiplier=4.)
    for arm in api.ARMS:
        kw=dict(arm=arm,seed=17,settings=settings,identity=dict(synthetic=True))
        a=api.fit(data,p,path=tmp_path/(arm+'a.pt.gz'),**kw)
        api.fit(data,p,path=tmp_path/(arm+'b.pt.gz'),stop_at=2,**kw)
        b=api.fit(data,p,path=tmp_path/(arm+'b.pt.gz'),resume=True,**kw)
        api.exact_refit(a,b)
        np.testing.assert_array_equal(api.predict(a,c,p),api.predict(b,c,p))
        assert a['round']==4 and len(a['trees'])==4
        with pytest.raises(ValueError):api.predict(a,dict(c,future_target=np.zeros((len(p),12,2))),p)
        with pytest.raises(ValueError):api.fit(data,p,path=tmp_path/(arm+'a.pt.gz'),**kw)
    with pytest.raises(ValueError):api.fit(dict(data,causal=dict(c,allowed=np.zeros_like(c['allowed']))),p,
        path=tmp_path/'empty.pt.gz',**kw)
