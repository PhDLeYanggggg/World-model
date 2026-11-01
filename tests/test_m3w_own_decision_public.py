import builtins
import copy
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import pytest
from scripts import report_m3w_own_decision_cost as report
from tests.test_m3w_active_bank_readout import fixture

run=report.run


def group(source='s0'):
    packet=fixture(source,missing=True)[0]['new_neural']
    p=np.broadcast_to([.1,.001,1.,.5,.0001],(2,7,5)).copy()
    choices={a:np.full(2,6) for a in (*run.api.NAMES,'frozen_projected_joint')}
    info=dict(queries=2,solver_calls=0,reason_counts={},nonoptimal={},same_count_controls_every_query=True)
    result=run.evaluation.readout(packet,choices,{a:np.ones(2) for a in choices},info,
        {a:p for a in run.api.ARMS},1.5,np.zeros((2,2)),np.ones(5))
    return result,{a:run.previous.coherence(p,packet['allowed'],packet['scales']) for a in run.api.ARMS}


def test_full_roster_guard_and_positive_known_result_cannot_hide_unknown_risk():
    contexts={f'c{i}':[f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    rows=[]
    for c,ss in contexts.items():
        for s in ss:
            result,coherence=group(s)
            for seed in (17,29,43):rows.append(dict(context=c,source=s,seed=seed,result=copy.deepcopy(result),coherence=coherence,
                matched_training=True,original_control_exact=True,minimum_gradient_query_visits=4))
    cfg=dict(seeds=[17,29,43],bootstrap_seed=4,bootstrap_resamples=32)
    summary=run.evaluation.summarize(rows,contexts,cfg)
    assert summary['accuracy']['own_joint']['point']>0
    assert summary['safety']['own_joint']['selected_unknown']>0
    assert not summary['fitting_advancement'] and not summary['independent_confirmation'] and not summary['deployment_changed']
    assert not summary['stage5c_executed'] and not summary['smc_enabled']
    with pytest.raises(ValueError):run.evaluation.summarize(rows[:-1],contexts,cfg)
    with pytest.raises(ValueError):run.evaluation.summarize(rows[:-1]+rows[:1],contexts,cfg)


def test_cold_output_directory_figure(tmp_path,monkeypatch):
    public=tmp_path/'public';private=tmp_path/'absent'
    monkeypatch.setattr(run,'PUBLIC',public);monkeypatch.setattr(run,'PRIVATE',private)
    interval=dict(point=1.,CI95=[.5,1.5])
    summary=dict(accuracy={a:interval for a in ('frozen_projected_joint','fixed_joint','own_joint')},
        effects=dict(own_minus_fixed=dict(full={k:interval for k in ('known_pp','lower_pp')})))
    report.figure(dict(summary=summary))
    assert (private/'own_decision_readout.png').exists() and (public/'figures/own_decision_readout.svg').exists()


def test_actual_full_report_uses_no_private_inputs_when_complete(monkeypatch):
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Full real report not yet complete')
    po,fo,ioo=Path.open,builtins.open,io.open
    def guard(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).absolute()
            if any(p.is_relative_to(run.ROOT/name) for name in ('data','external_data')):raise AssertionError('Public report must not read private inputs')
    def path_open(path,*a,**kw):guard(path);return po(path,*a,**kw)
    def file_open(path,*a,**kw):guard(path);return fo(path,*a,**kw)
    def io_open(path,*a,**kw):guard(path);return ioo(path,*a,**kw)
    monkeypatch.setattr(Path,'open',path_open);monkeypatch.setattr(builtins,'open',file_open);monkeypatch.setattr(io,'open',io_open)
    value=report.build();assert value['summary']['groups']==72
    assert value['training']['freeze']['primary_optimizer_updates']==208320


def test_complete_bundle_has_only_public_scalar_groups():
    path=run.PUBLIC/'metric_groups.zip'
    if not path.exists():pytest.skip('Full scalar archive not yet complete')
    forbidden={'features','targets','rollouts','floor_rollout','history_x','history_y','current_x','current_y','future_endpoint','model','optimizer'}
    def inspect(x):
        if isinstance(x,dict):
            assert not set(x)&forbidden
            for v in x.values():inspect(v)
        elif isinstance(x,list):
            for v in x:inspect(v)
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==72
        for name in z.namelist():
            assert name.startswith('groups/') and name.endswith('.json')
            raw=z.read(name);assert b'/Users/' not in raw
            inspect(json.loads(raw,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v))))
