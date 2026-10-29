import copy
import numpy as np
import pytest
from src.evaluation import m3w_detector_cost_readout as api


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
