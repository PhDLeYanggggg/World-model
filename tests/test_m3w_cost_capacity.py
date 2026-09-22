import numpy as np
import pytest
import torch
from src.world_model.m3w_tempered_cost_head import fit
from src.world_model.m3w_cost_capacity import fork_continuation, fit_path, save_prefix
from src.world_model.m3w_native_gain_harm import preprocess
from src.evaluation.m3w_experiment_contract import file_digest


def fixture():
    rng = np.random.default_rng(6); x = rng.normal(size=(40,5)).astype(np.float32)
    d = rng.uniform(.1,5,40); y = d[:,None]*rng.uniform(0,.3,(40,2)); sites = np.repeat(['a','b'],20)
    cv = np.ones(40); y[0],cv[0] = np.nan,np.nan
    pr = preprocess(x,y,cv,sites,'c')
    st = dict(width=8,steps=12,batch_size=16,learning_rate=.001,gradient_clip=5.,checkpoint_every=4,heartbeat_every=4)
    return x,y,d,sites,pr,st


def test_continuation_keeps_weights_rng_parent_and_exact_uninterrupted_result(tmp_path):
    x,y,d,sites,pr,st = fixture(); kw=dict(seed=17,heartbeat=lambda **v:None)
    fit(x,y,d,sites,pr,settings=st|{'steps':4},identity={'old':True},directory=tmp_path/'old',**kw)
    old=tmp_path/'old/checkpoint.pt'; sha=file_digest(old)
    r=fit_path(x,y,d,sites,pr,width='narrow',source=old,settings=st,identity={'new':True},
        directory=tmp_path/'new',stop_at=8,prefix_steps=4,**kw)
    assert not r['complete'] and r['newly_trained_updates']==4
    r=fit_path(x,y,d,sites,pr,width='narrow',source=old,settings=st,identity={'new':True},
        directory=tmp_path/'new',resume=True,prefix_steps=4,**kw)
    fit(x,y,d,sites,pr,settings=st,identity={'full':True},directory=tmp_path/'full',**kw)
    a=torch.load(tmp_path/'new/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    for key in a['model']: assert torch.equal(a['model'][key],b['model'][key])
    np.testing.assert_array_equal(a['draws'],b['draws'])
    assert file_digest(old)==sha and r['complete'] and r['newly_trained_updates']==8
    with pytest.raises(ValueError):
        fork_continuation(old,tmp_path/'bad.pt',identity={},settings=st|{'learning_rate':.2},seed=17,prefix_steps=4)


def test_wide_prefix_and_sampler_match_and_missing_prefix_fails(tmp_path):
    x,y,d,sites,pr,st=fixture(); kw=dict(seed=17,heartbeat=lambda **v:None)
    fit(x,y,d,sites,pr,settings=st|{'steps':4},identity={'old':True},directory=tmp_path/'old',**kw)
    old=tmp_path/'old/checkpoint.pt'; settings=st|{'width':16}
    r=fit_path(x,y,d,sites,pr,width='wide',source=old,settings=settings,identity={'wide':True},
        directory=tmp_path/'wide',prefix_steps=4,**kw)
    short=torch.load(tmp_path/'wide/prefix.pt',weights_only=False)
    assert short['step']==4 and r['step']==12 and r['newly_trained_updates']==12
    assert short['settings']['steps']==12
    with pytest.raises(ValueError): save_prefix(tmp_path/'wide/checkpoint.pt',tmp_path/'missing.pt',steps=4,identity={'wide':True})
    with pytest.raises(ValueError):
        fit_path(x,y,d,sites,pr,width='wide',source=old,settings=settings,identity={'wide':True},
            directory=tmp_path/'wide',prefix_steps=4,**kw)
