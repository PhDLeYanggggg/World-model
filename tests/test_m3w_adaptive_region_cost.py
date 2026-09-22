import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.run_m3w_adaptive_region_cost import validate_config
from src.world_model.m3w_adaptive_region_cost_head import fit, refreshed_region
from src.world_model.m3w_conditional_cost_head import fit as frozen_fit, region_weights
from src.world_model.m3w_bounded_cost_head import predict
from src.world_model.m3w_native_gain_harm import preprocess


def fixture():
    rng=np.random.default_rng(27)
    x=rng.normal(size=(40,5)).astype(np.float32);d=rng.uniform(.1,5,40)
    y=d[:,None]*rng.uniform(0,.3,(40,2));cv=np.ones(40)
    sites=np.repeat(['a','b'],20);y[0],cv[0]=np.nan,np.nan
    past=np.cumsum(rng.normal(size=(40,8,2)),axis=1).astype(np.float32)
    pr=preprocess(x,y,cv,sites,'c');w=region_weights(np.arange(40)%3==0,pr)
    settings=dict(width=8,learning_rate=.001,steps=12,batch_size=16,
        gradient_clip=5.,heartbeat_every=4,checkpoint_every=4)
    return x,y,d,past,sites,pr,w,settings


@pytest.mark.parametrize('boundary',[4,5,8,12])
def test_refresh_resume_draws_and_first_block_match(tmp_path,boundary):
    x,y,d,past,sites,pr,w,settings=fixture()
    kw=dict(seed=17,settings=settings,identity={'synthetic':True},heartbeat=lambda **v:None)
    dyn=dict(refresh_every=4,multiplier=4.)
    m,result=fit(x,y,d,past,sites,pr,w,directory=tmp_path/'full',**kw,**dyn)
    fit(x,y,d,past,sites,pr,w,directory=tmp_path/'resume',stop_at=boundary,**kw,**dyn)
    mr,_=fit(x,y,d,past,sites,pr,w,directory=tmp_path/'resume',resume=True,**kw,**dyn)
    mf,_=frozen_fit(x,y,d,sites,pr,w,directory=tmp_path/'frozen',stop_at=4,**kw)
    cp=torch.load(tmp_path/'full'/'checkpoint.pt',weights_only=False)
    rp=torch.load(tmp_path/'resume'/'checkpoint.pt',weights_only=False)
    fp=torch.load(tmp_path/'frozen'/'checkpoint.pt',weights_only=False)
    assert [r['step'] for r in cp['refreshes']]==[4,8]
    for k,v in mf.state_dict().items():
        assert torch.equal(v,cp['refreshes'][0]['model'][k])
    np.testing.assert_array_equal(predict(m,x,d,pr,'bounded_native'),predict(mr,x,d,pr,'bounded_native'))
    np.testing.assert_array_equal(cp['draws'],rp['draws'])
    frozen_fit(x,y,d,sites,pr,w,directory=tmp_path/'frozen',resume=True,**kw)
    fp=torch.load(tmp_path/'frozen'/'checkpoint.pt',weights_only=False)
    np.testing.assert_array_equal(cp['draws'],fp['draws'])
    for r,rr in zip(cp['refreshes'],rp['refreshes']):
        np.testing.assert_array_equal(r['selected'],rr['selected'])
        assert r['mean_weight']==pytest.approx(1.)
    assert result['refresh_count']==2 and result['unknown_rows_sampled']==0
    np.testing.assert_array_equal(cp['loss_weights'],region_weights(cp['refreshes'][-1]['selected'],pr,4.))
    with pytest.raises(ValueError,match='Changed resume'):
        fit(x,y,d,past,sites,pr,w,directory=tmp_path/'resume',resume=True,
            refresh_every=3,multiplier=4.,**kw)


def test_refresh_uses_no_future_target_and_preserves_rng(tmp_path):
    x,y,d,past,sites,pr,w,settings=fixture()
    m,_=frozen_fit(x,y,d,sites,pr,w,seed=17,settings=settings,
        identity={'synthetic':True},directory=tmp_path,heartbeat=lambda **v:None,stop_at=4)
    assert list(inspect.signature(refreshed_region).parameters)==['model','x','distance','past','pr','multiplier']
    m.train();rng=torch.get_rng_state().clone()
    bits,weights=refreshed_region(m,x,d,past,pr,4.)
    assert m.training and torch.equal(torch.get_rng_state(),rng)
    assert weights[~pr['known']].sum()==0 and np.dot(pr['weights'],weights)==pytest.approx(1.)
    stopped=np.repeat(past[:,-1:, :],8,axis=1)
    b,_=refreshed_region(m,x,d,stopped,pr,4.)
    assert not b.any()


def test_registered_budget_and_only_refresh_change():
    cfg=json.loads(Path('configs/m3w_adaptive_region_cost_v1.json').read_text())
    pcfg=json.loads(Path('configs/m3w_conditional_cost_v1.json').read_text())
    validate_config(cfg,pcfg)
    assert len(range(cfg['refresh_every'],cfg['training']['steps'],cfg['refresh_every']))==23
    for k,v in [('refresh_every',250),('region_multiplier',8.),('risk_calibration',True),
        ('primary_reference','tempered_strict_stop'),('training',cfg['training']|{'steps':24000}),
        ('threshold_search',True),('closed_role_readout',True)]:
        with pytest.raises(ValueError):validate_config(cfg|{k:v},pcfg)
