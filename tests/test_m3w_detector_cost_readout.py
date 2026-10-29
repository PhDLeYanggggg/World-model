import copy
import numpy as np
import pytest
import torch
from src.evaluation import m3w_detector_cost_readout as api
from src.world_model import m3w_detector_cost as model


def population():
    contexts={f'c{i}':[dict(packet=dict(identity=dict(source=f's{j}')))
        for j in range((i%3)*4,(i%3)*4+4)] for i in range(6)}
    cfg=dict(seeds=[17,29,43],bootstrap_seed=11,bootstrap_resamples=32)
    y=np.broadcast_to([.1,0.,1.,1.,0.],(3,7,5)).copy();y[-1]=np.nan
    score=api.base.compact_score(y,np.ones((3,7)),np.array([0,-1,-1]))
    bounds={m:dict(known_pp=1.,lower_pp=.5,upper_pp=1.5) for m in ('full','matched')}
    rows=[]
    for context,source,seed,arm in sorted(api.expected_roster(contexts,cfg['seeds'])):
        policies={b:dict(policy=copy.deepcopy(score),known_ADE_gain_floor_percent=1.,known_FDE_gain_floor_percent=1.) for b in api.BANKS}
        subset=dict(reference_ADE_mass=1.,reference_FDE_mass=1.,known_rows=2,known_FDE_rows=2,
            policies={b:dict(ADE_error_mass=.9,FDE_error_mass=.9,switched=1) for b in api.BANKS})
        result=dict(policies=policies,recordings={'record':{b:copy.deepcopy(score) for b in api.BANKS}},
            representation_contrasts={c:{b:copy.deepcopy(bounds) for b in api.BANKS} for c in api.REPRESENTATIONS},
            comparisons={'physical_neural_minus_physical_only':copy.deepcopy(bounds)},
            subsets={s:copy.deepcopy(subset) for s in ('hard','easy')})
        rows.append(dict(context=context,source=source,seed=seed,representation=arm,result=result))
    return rows,contexts,cfg,y


def test_complete_fitting_pass_cannot_be_called_independent_or_deployable():
    rows,contexts,cfg,_=population();out=api.summarize(rows,contexts,cfg)
    assert out['fitting_advancement_gate'] and out['heads']==144
    assert not out['deployment_changed'] and not out['submission_ready'] and not out['independent_confirmation']
    assert out['representations']['history']['accuracy']['physical_neural']['bootstrap_resamples']==32


def test_missing_duplicate_and_empty_risk_never_disappear():
    rows,contexts,cfg,y=population()
    with pytest.raises(ValueError):api.summarize(rows[:-1],contexts,cfg)
    with pytest.raises(ValueError):api.summarize(rows[:-1]+rows[:1],contexts,cfg)
    row=next(r for r in rows if r['representation']=='history')
    empty=api.base.compact_score(y,np.ones((3,7)),np.full(3,-1,int))
    row['result']['recordings']['record']['physical_neural']=empty
    out=api.summarize(rows,contexts,cfg)
    assert not out['fitting_advancement_gate']
    assert out['representations']['history']['safety']['physical_neural']['undefined_easy_record_views']==1


def test_each_information_control_and_neural_increment_must_pass():
    rows,contexts,cfg,_=population()
    for row in rows:
        if row['representation']=='history':
            row['result']['representation_contrasts']['noise']['physical_neural']['matched']['lower_pp']=-1.
    out=api.summarize(rows,contexts,cfg)
    assert not out['fitting_advancement_gate']
    assert not out['fitting_gates']['history_minus_noise_matched_lower_CI_positive']


def test_disabled_inference_matches_original_and_active_inference_keeps_cost_bounds():
    torch.set_num_threads(1);rng=np.random.default_rng(8);n=9
    x=rng.normal(size=(n,7,380));scene=rng.normal(size=(n,7,18));extra=rng.normal(size=(n,64))
    scales=rng.uniform(.5,2,n);envelope=rng.uniform(0,2,(n,7))
    pr=dict(mean=np.zeros(398),std=np.ones(398),clip=8.,bias=np.zeros(5,np.float32),
        detector_normalizer=dict(mean=np.zeros(64),std=np.ones(64),clip=8.),detector_representation='disabled')
    head=model.DetectorCostHead(462,64,17,pr['bias'],representation='disabled')
    state=dict(model=head.state_dict(),preprocess=pr,settings=dict(width=64),seed=17,dimension=462,mode='known_only')
    old=dict(state,model={k:v for k,v in state['model'].items() if k!='detector_weight'},dimension=398)
    prediction=model.predict(state,x,scene,extra,envelope,scales)
    np.testing.assert_array_equal(prediction,model.parent.predict(old,x,scene,envelope,scales))
    with torch.no_grad():head.detector_weight.fill_(.03)
    state['preprocess']=dict(pr,detector_representation='history');state['model']=head.state_dict()
    prediction=model.predict(state,x,scene,extra,envelope,scales)
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    assert (prediction[...,0]<=prediction[...,2]+1e-6).all()
    assert (prediction[...,3]<=prediction[...,2]+1e-6).all()
    assert (prediction[...,4]<=prediction[...,1]+1e-6).all()
    assert (prediction[...,0]+prediction[...,1]<=envelope+1e-6).all()
    with pytest.raises(ValueError):model.predict(state,x,scene,extra,-envelope,scales)
