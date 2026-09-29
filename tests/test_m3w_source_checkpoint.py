import numpy as np
import pytest
from src.world_model import m3w_source_checkpoint as api


def test_recording_split_disjoint_and_order_invariant():
    r=np.repeat(['a','b','c'],10);f=np.tile(np.arange(10),3)
    a,b,d=api.source_partition(r,f,'site')
    assert set(r[a]).isdisjoint(r[b]) and d['purged_rows']==0
    aa,bb,_=api.source_partition(r[::-1],f[::-1],'site')
    np.testing.assert_array_equal(a,aa[::-1]);np.testing.assert_array_equal(b,bb[::-1])


def test_single_recording_full_label_footprint_purged():
    f=np.arange(0,2400,12);a,b,d=api.source_partition(np.repeat('r',len(f)),f,'site')
    assert f[a].max()+144<f[b].min()-84 and d['purged_rows']>0
    with pytest.raises(ValueError):api.source_partition(np.repeat('r',4),np.arange(4),'site')


def fixture():
    rng=np.random.default_rng(7);n=90
    x=rng.normal(size=(n,4));env=np.ones(n)
    y=np.tile([.2,.1,1.,.6,.08],(n,1));r=np.repeat(['a','b','c'],30);f=np.tile(np.arange(30),3)
    return x,env,y,np.repeat('site',n),r,f


def test_resume_and_validation_only_checkpoint_selection(tmp_path):
    api.torch.set_num_threads(1)
    x,env,y,s,r,f=fixture()
    settings=dict(steps=4,width=4,learning_rate=.001,query_batch_size=2,gradient_clip=5.,validate_every=2,checkpoint_every=2)
    kwargs=dict(source='site',settings=settings,seed=17,identity={'test':True},heartbeat=lambda **kw:None)
    direct=api.fit(x,env,y,s,r,f,path=tmp_path/'direct.pt',**kwargs)
    api.fit(x,env,y,s,r,f,path=tmp_path/'resume.pt',stop_at=2,**kwargs)
    resumed=api.fit(x,env,y,s,r,f,path=tmp_path/'resume.pt',resume=True,**kwargs)
    for k in ('model','optimizer','best_model','best_step','best_score','trace','sampler_rng','draw_hash'):api.core.exact(direct[k],resumed[k])
    best=min(direct['trace'],key=lambda z:z['validation_signed_MSE'])
    assert direct['best_step']==best['step']
    a,_,_=api.source_partition(r,f,'site')
    pr=api.core.preprocess(x[a],env[a],y[a],s[a],r[a],f[a],training_site='site')
    api.core.exact(pr,direct['preprocess'])


def test_source_mixing_rejected(tmp_path):
    x,env,y,s,r,f=fixture();s[0]='bad'
    with pytest.raises(ValueError):
        api.fit(x,env,y,s,r,f,source='site',settings={},seed=1,path=tmp_path/'bad.pt',identity={},heartbeat=lambda **kw:None)


def test_empty_matched_interventions_not_invented():
    p=np.tile([.01,.1,1.,.5,.08],(3,1))
    d=api.decisions({'final':p,'validation':p},np.ones(3,bool),np.ones(3,bool),np.zeros(3,int),np.zeros(3,int),np.arange(3))
    assert all(not z.any() for z in d.values())
