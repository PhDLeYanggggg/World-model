import numpy as np
import pytest
import torch
from src.world_model import m3w_query_excess as api
from scripts.replay_m3w_dimensionless_training import exact


def sample():
    rng=np.random.default_rng(4); x=rng.normal(size=(40,5)).astype(np.float32)
    u=rng.normal(size=(40,6)).astype(np.float32)
    y=np.abs(rng.normal(size=(40,4))); y[:,[1,3]]*=.01
    sites=np.array(['a']*20+['b']*20); env=np.ones(40)*3
    known=np.ones(40,bool);known[-1]=False;y[-1]=np.nan
    w=np.zeros(40);w[:20]=.5/20;w[20:39]=.5/19
    pr=dict(mean=x.mean(0),std=x.std(0),cost_scale=1.,clip=10.,known=known,
        weights=w,constant=(y[known]*w[known,None]).sum(0),training_sites=['a','b'])
    return x,u,y,sites,np.repeat('r',40),np.tile(np.repeat(np.arange(5),4),2),env,pr


def test_query_grouping_never_crosses_source_recording_or_frame():
    groups,sg,keys=api.query_groups(np.array(['a','a','a','b']),np.array(['r','r','s','r']),
        np.array([1,2,1,1]),np.ones(4,bool))
    assert len(groups)==4 and [len(x) for x in sg]==[3,1]
    assert len(set(keys))==4


def test_unknown_labels_not_drawn_source_sampling_balanced():
    *_,sites,records,frames,env,pr=sample()
    groups,sg,_=api.query_groups(sites,records,frames,pr['known'])
    ix,seg,qids=api.draw_queries(groups,sg,8,torch.Generator().manual_seed(2))
    assert 39 not in ix and len(set(seg))==8
    assert sum(i in sg[0] for i in qids)==4


def test_aggregate_loss_distinguishes_canceling_errors_and_equal_query_weights():
    target=torch.zeros((3,4)); pred=torch.zeros((3,4),requires_grad=True)
    with torch.no_grad():pred[:,1]=torch.tensor([1.,-1.,2.])
    value=api.losses(pred,target,torch.tensor([0,0,1]),2)
    assert float(value['pointwise'].detach())==pytest.approx(1.25)
    assert float(value['query'].detach())==pytest.approx(1.)
    value['query'].backward();assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize('arm',['pointwise','query'])
def test_resume_exact_and_unknown_supervision_excluded(tmp_path,arm):
    values=sample();settings=dict(width=8,steps=5,query_batch_size=4,learning_rate=.0003,
        gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    args=dict(arm=arm,seed=17,settings=settings,identity={'test':True},heartbeat=lambda **kw:None)
    model,info=api.fit(*values,directory=tmp_path/'full',**args)
    api.fit(*values,directory=tmp_path/'split',stop_at=2,**args)
    other,_=api.fit(*values,directory=tmp_path/'split',resume=True,**args)
    assert info['unknown_rows_sampled']==0
    a,b=[api.head.read_checkpoint(tmp_path/p/'checkpoint.pt.gz') for p in ('full','split')]
    for k in ('model','optimizer','sampler_rng','torch_rng','draws','query_draws','trace'):exact(a[k],b[k])
    exact(model.state_dict(),other.state_dict())
    with pytest.raises(ValueError):api.fit(*values,directory=tmp_path/'full',**args)


def test_matched_arms_share_initialization_queries_rows_and_runtime_rng(tmp_path):
    values=sample();settings=dict(width=8,steps=3,query_batch_size=4,learning_rate=.0003,
        gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    for arm in ('pointwise','query'):
        api.fit(*values,arm=arm,seed=17,settings=settings,identity={'test':True},
            directory=tmp_path/arm,heartbeat=lambda **kw:None)
    api.assert_matched(*[api.head.read_checkpoint(tmp_path/a/'checkpoint.pt.gz') for a in ('pointwise','query')])


def test_prediction_api_excludes_future_and_target_fields():
    import inspect
    assert not {'target','future','valid','future_mask'} & set(inspect.signature(api.head.predict).parameters)


@pytest.mark.parametrize('path',['src/world_model/m3w_query_excess.py','scripts/run_m3w_european_query_excess_refit.py'])
def test_rosetta_rejected_before_any_torch_import(monkeypatch,path):
    import builtins
    import platform
    import runpy
    monkeypatch.setattr(platform,'system',lambda:'Darwin')
    monkeypatch.setattr(platform,'machine',lambda:'x86_64')
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name!='torch','Torch imported before architecture guard'
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(RuntimeError,match='Native arm64'):
        runpy.run_path(path,run_name='__main__')
