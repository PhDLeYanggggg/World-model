import numpy as np
import pytest
import torch
from src.world_model import m3w_fixed_floor_tail as api


def test_positive_tail_uses_training_weights_and_no_positive_fallback():
    y=np.array([[1,0,1,0],[1,1,1,0],[1,2,1,0],[1,9,1,0],[np.nan]*4])
    w=np.array([.25,.25,.25,.25,0]); t=api.tail_parameters(y,w)
    assert t['thresholds']==[9.,None]
    weights=api.tail_weights(torch.tensor(y[:4]),t).numpy()
    np.testing.assert_allclose(weights[:,1],[1/1.75]*3+[4/1.75])
    np.testing.assert_array_equal(weights[:,[0,2,3]],np.ones((4,3)))
    assert np.isclose(np.mean(weights[:,1]),1.)


def test_tail_loss_does_not_backpropagate_targets_or_reweight_references():
    p=torch.ones(3,4,requires_grad=True)
    y=torch.tensor([[2.,0,2,0],[2,1,2,1],[2,4,2,4]],requires_grad=True)
    t=api.tail_parameters(y.detach().numpy(),np.ones(3)/3)
    wm=api.tail_weights(y,t)
    assert not wm.requires_grad
    api.loss(p,y,'tail4',t).backward()
    assert y.grad is None
    np.testing.assert_allclose(p.grad[:,0].numpy(),p.grad[:,2].numpy())
    assert p.grad[-1,1].abs()>p.grad[0,1].abs()


def test_head_nonnegative_harm_bounded_without_posthoc_clipping():
    pr=dict(mean=np.zeros(4),constant=np.array([4.,.1,2.,.02]),cost_scale=2.)
    head=api.initialize(pr,8,17,1.)
    p=head(torch.zeros(5,4),torch.tensor([0.,1.,2.,3.,4.]))
    assert (p[:,[0,2]]>0).all() and (p[:,[1,3]]>=0).all()
    assert (p[:,1]<=torch.arange(5.)).all() and (p[:,3]<=torch.arange(5.)).all()


def test_match_counts_are_current_query_only_and_keep_unknown_rows():
    key=np.array([10,10,11,11,12]); frames=np.array([1,2,1,1,1]); ids=np.arange(5)
    anchor=np.array([True,False,False,True,False]); eligible=np.ones(5,bool)
    score=np.array([99,-99,-2,2,-9.])
    mask=api.match_counts(anchor,eligible,score,key,frames,ids)
    assert mask.tolist()==[True,False,True,False,False]
    # Frame2's much better score cannot replace the current frame1 action.
    with pytest.raises(ValueError): api.match_counts(anchor,np.zeros(5,bool),score,key,frames,ids)


def test_tail_train_resume_and_matched_sampler(tmp_path):
    from scripts.replay_m3w_dimensionless_training import exact
    torch.set_num_threads(4); rng=np.random.default_rng(7)
    x=rng.normal(size=(40,6)).astype(np.float32); sites=np.repeat(['a','b'],20)
    y=np.column_stack((np.ones(40),np.linspace(0,.8,40),np.ones(40)*.5,np.linspace(0,.2,40)))
    y[[0,20]]=np.nan; known=np.isfinite(y).all(1); w=known/known.sum()
    pr=dict(mean=x[known].mean(0),std=x[known].std(0),cost_scale=1.,known=known,weights=w,
            constant=(y[known]*w[known,None]).sum(0),clip=10.,training_sites=['a','b'])
    settings=dict(width=8,steps=8,batch_size=16,learning_rate=.0003,gradient_clip=5.,checkpoint_every=4,heartbeat_every=4)
    kwargs=dict(seed=17,settings=settings,identity={'fixed':True},heartbeat=lambda **kw:None)
    env=np.ones(40)
    a,_=api.fit(x,y,sites,env,pr,arm='tail4',directory=tmp_path/'a',**kwargs)
    api.fit(x,y,sites,env,pr,arm='tail4',directory=tmp_path/'b',stop_at=4,**kwargs)
    b,br=api.fit(x,y,sites,env,pr,arm='tail4',directory=tmp_path/'b',resume=True,**kwargs)
    exact(a.state_dict(),b.state_dict()); assert br['unknown_rows_sampled']==0
    sa=torch.load(tmp_path/'a/checkpoint.pt',weights_only=False)
    sb=torch.load(tmp_path/'b/checkpoint.pt',weights_only=False)
    for k in ('draws','trace','optimizer','sampler_rng'): exact(sa[k],sb[k])
    api.fit(x,y,sites,env,pr,arm='mse',directory=tmp_path/'c',**kwargs)
    sc=torch.load(tmp_path/'c/checkpoint.pt',weights_only=False)
    for k in ('draws','sampler_rng','preprocess','tail','initial_model'): exact(sa[k],sc[k])
    with pytest.raises(ValueError): api.fit(x,y,sites,env,pr,arm='tail4',directory=tmp_path/'a',**kwargs)
