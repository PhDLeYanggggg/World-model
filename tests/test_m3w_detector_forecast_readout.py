import copy
import json
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
    record=next(iter(next(iter(rows[0]['result']['sites'].values()))['records'].values()))
    assert record['reference_ADE']==dict(known_rows=1,mean=1.,p95=1.)
    assert set(record['physical_ADE'])=={str(k) for k in range(6)}
    with pytest.raises(ValueError):api.summarize(rows[:-1],jobs,cfg)
    with pytest.raises(ValueError):api.summarize(rows[:-1]+rows[:1],jobs,cfg)


def test_even_positive_proposal_input_gate_is_not_learned_policy_safety():
    rows,jobs,cfg=population(unknown=False);out=api.summarize(rows,jobs,cfg)
    assert out['proposal_input_evidence_gate']
    assert not out['learned_policy_safety_evaluated'] and not out['submission_ready']
    assert not out['independent_roles_opened'] and not out['stage5c_executed']
    physical=out['kinds']['single']['physical_baselines']
    assert set(physical)=={str(k) for k in range(6)}
    assert all(v['ADE']['point']==0. for v in physical.values())


def test_producer_locality_or_seed_mismatch_cannot_be_relabelled():
    rows,jobs,cfg=population();rows[0]['seed']=99
    with pytest.raises(ValueError):api.summarize(rows,jobs,cfg)


def test_report_keeps_all_baselines_and_undefined_views_explicit():
    from scripts import report_m3w_detector_forecaster as report
    rows,jobs,cfg=population();summary=api.summarize(rows,jobs,cfg)
    value=dict(training=dict(pid=1,seconds=1.,peak_RSS_bytes=1),
        readout=dict(pid=2,seconds=1.),replay=dict(pid=3,seconds=1.),
        loss_summary={},support_profile={},proposal_gates=summary['proposal_input_gates'],
        proposal_input_evidence_gate=False)
    text=report.render(value,summary)
    assert 'not_estimable' in text
    assert all(name in text for name in report.run.source.BASELINES)
    assert 'learned_policy_safety_evaluated: false' in text
    assert 'deployment_changed: false' in text


def test_empty_record_tail_is_not_zero_risk():
    assert api.tail_stats(np.array([np.nan,2.]),np.array([True,False]))==dict(known_rows=0,mean=None,p95=None)


def test_support_profile_counts_repeated_views_without_multiplying_by_arms(tmp_path,monkeypatch):
    from scripts import report_m3w_detector_forecaster as report
    monkeypatch.setattr(report,'ROOT',tmp_path)
    rows,_,_=population();refs=[]
    for i,row in enumerate(rows):
        path=tmp_path/(str(i)+'.json');path.write_text(json.dumps(row))
        refs.append(dict(path=path.name,sha256=report.run.sha(path)))
    out=report.support_profile(dict(groups=refs))
    assert out['single']['indexed_row_views']==144
    assert out['single']['missing_future_row_views']==72
    assert out['single']['complete_future_row_views']==72
    assert out['single']['unique_recordings']==12
    assert out['complement']['indexed_row_views']==72
    with pytest.raises(ValueError):report.support_profile(dict(groups=refs[:-1]))
    with pytest.raises(ValueError):report.support_profile(dict(groups=refs[:-1]+refs[:1]))
    (tmp_path/refs[0]['path']).write_text('{}')
    with pytest.raises(ValueError):report.support_profile(dict(groups=refs))
