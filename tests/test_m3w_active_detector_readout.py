import copy
import numpy as np
import pytest
from src.evaluation import m3w_active_detector_readout as api


def example(missing=True):
    sites=np.repeat([f's{i}' for i in range(12)],2);n=len(sites);records=np.repeat(np.arange(12),2)
    target=np.zeros((n,12,2));valid=np.ones((n,12),bool)
    if missing:valid[1::2]=False;target[1::2]=1e100
    physical=np.zeros((n,6,12,2));physical[...,0]=1.
    predictions={a:np.zeros((n,12,2)) for a in api.ARMS}
    for a,v in dict(disabled=1.,current=.8,noise=.9,history=.5).items():predictions[a][...,0]=v
    rows=[];jobs=[]
    for fold in range(3):
        fit=[f's{i}' for i in range(12) if i%3==fold];use=~np.isin(sites,fit)
        result=api.site_readout(np.arange(n)[use],sites[use],records[use],np.zeros((use.sum(),8,2)),target[use],valid[use],
            physical[use],1,{a:p[use] for a,p in predictions.items()},easy_cut=2.,hard_cut=3.)
        for seed in (17,29,43):
            key=f'single{fold}_seed{seed}';jobs.append(dict(key=key,identity=dict(kind='single',seed=seed,fit_sites=fit)))
            for arm in api.ARMS:rows.append(dict(trial=key,kind='single',seed=seed,representation=arm,
                result=copy.deepcopy(result),disabled_legacy_exact=True))
    return rows,jobs,dict(seeds=[17,29,43],bootstrap_seed=17,bootstrap_resamples=32)


def test_active_input_positive_known_gain_does_not_hide_missing_completion_or_deploy():
    rows,jobs,cfg=example();s=api.summarize(rows,jobs,cfg)
    assert s['history_contrasts']['all']['disabled']['known_pp']['point']==50.
    assert s['history_contrasts']['all']['disabled']['lower_pp']['point']==0.
    assert not s['proposal_input_evidence_gate'] and not s['deployment_changed']
    assert set(s['history_vs_fixed_physical']['all'])==set(map(str,range(6)))
    assert all(v['known_pp']['point']==50. and v['lower_pp']['point']==0. for v in s['history_vs_fixed_physical']['all'].values())
    assert s['representations']['history']['hard']['ADE']['point'] is None


def test_positive_raw_proposal_gate_never_certifies_complete_learned_policy():
    rows,jobs,cfg=example(missing=False);s=api.summarize(rows,jobs,cfg)
    assert s['proposal_input_evidence_gate'] and not s['learned_policy_safety_evaluated']
    assert not s['independent_roles_opened'] and not s['stage5c_executed'] and not s['smc_enabled']


def test_missing_duplicated_or_wrong_seed_parent_readouts_fail():
    rows,jobs,cfg=example()
    with pytest.raises(ValueError):api.summarize(rows[:-1],jobs,cfg)
    with pytest.raises(ValueError):api.summarize(rows[:-1]+rows[:1],jobs,cfg)
    rows[0]['seed']=99
    with pytest.raises(ValueError):api.summarize(rows,jobs,cfg)
