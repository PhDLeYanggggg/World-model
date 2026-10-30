import copy
import numpy as np
import pytest
from src.evaluation import m3w_detector_forecast_readout as api


def population(unknown=True):
    sites=np.array([f's{i}' for i in range(12) for _ in range(2)]);n=len(sites)
    records=np.repeat(np.arange(12),2);ids=np.arange(n);history=np.zeros((n,8,2));target=np.zeros((n,12,2))
    valid=np.ones((n,12),bool)
    if unknown:valid[1::2]=False;target[1::2]=1e100
    physical=np.zeros((n,6,12,2));physical[...,0]=1.
    predictions={a:np.zeros((n,12,2)) for a in api.ARMS}
    for a,v in dict(disabled=1.,current=.8,noise=.9,history=.5).items():predictions[a][...,0]=v
    jobs=[];rows=[]
    for kind in ('single','complement'):
        for fold in range(3):
            fit=[f's{i}' for i in range(12) if (i%3==fold)==(kind=='single')];take=~np.isin(sites,fit)
            result=api.site_readout(ids[take],sites[take],records[take],history[take],target[take],valid[take],
                physical[take],1,{a:v[take] for a,v in predictions.items()},easy_cut=2.,hard_cut=3.)
            for seed in (17,29,43):
                key=f'{kind}{fold}_seed{seed}';jobs.append(dict(key=key,identity=dict(kind=kind,seed=seed,fit_sites=fit)))
                for arm in api.ARMS:rows.append(dict(trial=key,kind=kind,seed=seed,representation=arm,
                    result=copy.deepcopy(result),disabled_legacy_exact=True))
    cfg=dict(seeds=[17,29,43],bootstrap_seed=1,bootstrap_resamples=32)
    return rows,jobs,cfg


def test_missing_outcomes_remain_in_contrast_and_no_source_is_discarded():
    rows,jobs,cfg=population();out=api.summarize(rows,jobs,cfg)
    p=out['kinds']['single']['history_contrasts']['all']['disabled']
    assert p['known_pp']['point']==50.
    assert p['lower_pp']['point']==0.
    assert not out['proposal_input_evidence_gate'] and not out['deployment_changed']
    assert out['kinds']['single']['representations']['history']['hard']['ADE']['point'] is None
    with pytest.raises(ValueError):api.summarize(rows[:-1],jobs,cfg)
    with pytest.raises(ValueError):api.summarize(rows[:-1]+rows[:1],jobs,cfg)


def test_even_positive_proposal_input_gate_is_not_learned_policy_safety():
    rows,jobs,cfg=population(unknown=False);out=api.summarize(rows,jobs,cfg)
    assert out['proposal_input_evidence_gate']
    assert not out['learned_policy_safety_evaluated'] and not out['submission_ready']
    assert not out['independent_roles_opened'] and not out['stage5c_executed']


def test_producer_locality_or_seed_mismatch_cannot_be_relabelled():
    rows,jobs,cfg=population();rows[0]['seed']=99
    with pytest.raises(ValueError):api.summarize(rows,jobs,cfg)
