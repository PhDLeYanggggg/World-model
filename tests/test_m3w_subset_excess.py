import inspect
import numpy as np
import pytest
import torch
from src.world_model import m3w_subset_excess as api
from tests.test_m3w_query_excess import sample


def test_masks_are_causal_grouped_and_tie_stable():
    sites=np.array(['a','a','a','a','b']);rec=np.array(['r','r','r','s','r']);frame=np.ones(5,int)
    ids=np.array([3,2,1,4,5]);moving=np.array([1,1,0,1,1],bool);teacher=np.ones(5,bool)
    m=api.subset_masks(sites,rec,frame,ids,moving,teacher,np.ones(5))
    np.testing.assert_array_equal(m[:,0],moving)
    np.testing.assert_array_equal(m[:,1],[False,True,False,True,True])
    np.testing.assert_array_equal(m[:,2],[True,False,False,False,False])
    assert not (m[:,1]&m[:,2]).any()
    np.testing.assert_array_equal(m[:,1]|m[:,2],moving)
    assert not {'future','target','known','valid'} & set(inspect.signature(api.subset_masks).parameters)


def test_subset_aggregate_retains_individual_anchor_when_errors_cancel():
    target=torch.zeros(2,4);target[:,1]=1
    prediction=target.clone();prediction[:,1]=torch.tensor([2.,0.]);prediction.requires_grad_()
    loss=api.losses(prediction,target,torch.tensor([0,0]),1,torch.ones(2,3,dtype=torch.bool))
    assert loss['auxiliary_aggregate'].item()==0
    assert loss['subset_aggregate'].item()==pytest.approx(.25)
    assert loss['subset_pointwise'].item()==pytest.approx(.5)
    loss['subset_aggregate'].backward();assert prediction.grad.abs().sum()>0


def test_empty_subset_not_nan_and_no_future_target_gradient():
    p=torch.ones(3,4,requires_grad=True);y=torch.zeros(3,4,requires_grad=True)
    losses=api.losses(p,y,torch.tensor([0,0,1]),2,torch.zeros(3,3,dtype=torch.bool))
    assert losses['empty_subset_fraction'].item()==1
    assert losses['subset_aggregate'].item()==losses['subset_pointwise'].item()
    losses['subset_aggregate'].backward();assert y.grad is None


@pytest.mark.parametrize('arm',api.ARMS)
def test_resume_and_subset_identity_exact(tmp_path,arm):
    values=sample();subsets=np.ones((40,3),bool);subsets[::2,1]=False
    settings=dict(width=8,steps=5,query_batch_size=4,learning_rate=.0003,gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    kwargs=dict(arm=arm,seed=17,settings=settings,identity={'test':True},heartbeat=lambda **kw:None)
    api.fit(*values,subsets,directory=tmp_path/'full',**kwargs)
    api.fit(*values,subsets,directory=tmp_path/'split',stop_at=2,**kwargs)
    _,info=api.fit(*values,subsets,directory=tmp_path/'split',resume=True,**kwargs)
    a,b=[api.head.read_checkpoint(tmp_path/p/'checkpoint.pt.gz') for p in ('full','split')]
    for k in ('model','optimizer','sampler_rng','torch_rng','draws','query_draws','trace','subsets'):api.exact(a[k],b[k])
    assert info['unknown_rows_sampled']==0
    changed=subsets.copy();changed[0,0]=False
    with pytest.raises(AssertionError):api.fit(*values,changed,directory=tmp_path/'split',resume=True,**kwargs)


def test_matched_arms_same_initialization_rows_and_subsets(tmp_path):
    values=sample();subsets=np.ones((40,3),bool)
    settings=dict(width=8,steps=3,query_batch_size=4,learning_rate=.0003,gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    for arm in api.ARMS:
        api.fit(*values,subsets,arm=arm,seed=17,settings=settings,identity={'test':True},directory=tmp_path/arm,heartbeat=lambda **kw:None)
    api.assert_matched(*[api.head.read_checkpoint(tmp_path/arm/'checkpoint.pt.gz') for arm in api.ARMS])


def test_quality_keeps_empty_support_separate_from_zero_error():
    p=np.ones((3,4));y=np.zeros((3,4));y[-1]=np.nan
    q=api.quality(p,y,np.repeat('r',3),np.ones(3),np.zeros((3,3),bool))
    assert q['controller_admission_all_MSE'] is None
    assert q['controller_admission_known_nonempty_fraction']==0


def test_unknown_rows_participate_in_causal_partition():
    args=(np.repeat('a',4),np.repeat('r',4),np.ones(4,int),np.arange(4),
          np.ones(4,bool),np.ones(4,bool),np.arange(4,dtype=float))
    m=api.subset_masks(*args)
    known=np.array([False,True,True,True])
    np.testing.assert_array_equal(m[known,1],[True,False,False])
    filtered=api.subset_masks(*[a[known] for a in args])
    assert not np.array_equal(m[known,1],filtered[:,1])


def test_fixed_denominator_does_not_redefine_selected_risk():
    from scripts.run_m3w_european_subset_excess import harm_accounting
    f=np.array([1.,2.,np.nan]);n=np.array([2.,0.,np.nan]);known=np.array([True,True,False])
    abstain=harm_accounting(f,n,known,np.zeros(3,bool))
    assert abstain['selected_reference_error']==0 and abstain['positive_harm_over_all_floor']==0
    selected=harm_accounting(f,n,known,np.ones(3,bool))
    assert selected['positive_harm_sum']==1 and selected['benefit_sum']==2
    assert selected['positive_harm_over_all_floor']==pytest.approx(1/3)
    assert harm_accounting(np.zeros(3),np.ones(3),known,np.ones(3,bool))['positive_harm_over_all_floor'] is None


@pytest.mark.parametrize('path',['src/world_model/m3w_subset_excess.py','scripts/run_m3w_european_subset_excess.py'])
def test_rosetta_rejected_before_torch(monkeypatch,path):
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
