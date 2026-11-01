import builtins
import copy
import io
import json
import os
from pathlib import Path
import zipfile

import pytest
from scripts import report_m3w_free_allocation_actor as report

run=report.run


def frozen_fits():
    refs=json.loads((run.PUBLIC/'training_freeze.json').read_text())['fits']
    return [json.loads((run.ROOT/r['path']).read_text()) for r in refs]


def test_registered_definition_and_all36_real_fits():
    reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():assert run.sha(run.ROOT/rel)==digest
    assert reg['new_actor_fits']==36 and reg['primary_optimizer_updates']==72000
    assert not reg['independent_roles_opened'] and not reg['own_outcomes_read']
    stats=report.training_statistics(frozen_fits())
    assert stats['complete_fits']==36 and stats['exact_refit_updates']==4000
    assert set(stats['arms'])==set(run.api.ARMS)
    for name in run.api.ARMS:
        refit=json.loads((run.PUBLIC/f'first_full_refit_{name}.json').read_text())
        assert refit['scientific_state_exact'] and refit['updates']==2000


@pytest.mark.parametrize('problem',('missing','duplicated','step','unknown_label','draws','features'))
def test_incomplete_or_unmatched_training_cannot_be_reported_complete(problem):
    fits=copy.deepcopy(frozen_fits())
    if problem=='missing':fits.pop()
    elif problem=='duplicated':fits[-1]=fits[0]
    elif problem=='step':fits[0]['step']=100
    elif problem=='unknown_label':fits[0]['unknown_rows_with_supervised_labels']=1
    elif problem=='draws':fits[0]['draw_hash']='changed'
    elif problem=='features':fits[0]['input_hashes']['self_context']='changed'
    with pytest.raises(ValueError):report.training_statistics(fits)


def test_complete_actual_readout_if_available():
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Complete score verification not ready')
    v=report.build();s=v['summary']
    assert s['groups']==72 and s['unique_target_rows']==149510 and s['unique_recordings']==74
    assert not s['independent_confirmation'] and not s['whole_pipeline_OOF']
    assert not s['deployment_changed'] and not s['submission_ready']
    assert not v['replay']['optimization_replayed']
    assert len(s['fitting_gates'])==8


def test_report_build_uses_no_private_data(monkeypatch):
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Complete public evidence not ready')
    path_open=Path.open;file_open=builtins.open;io_open=io.open
    def guard(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).absolute()
            if any(p.is_relative_to(run.ROOT/k) for k in ('data','external_data')):
                raise AssertionError('Report must read public aggregates only')
    def p_open(path,*a,**kw):guard(path);return path_open(path,*a,**kw)
    def f_open(path,*a,**kw):guard(path);return file_open(path,*a,**kw)
    def i_open(path,*a,**kw):guard(path);return io_open(path,*a,**kw)
    monkeypatch.setattr(Path,'open',p_open);monkeypatch.setattr(builtins,'open',f_open);monkeypatch.setattr(io,'open',i_open)
    assert report.build()['summary']['groups']==72


def test_metric_bundle_is_scalar_only_if_available():
    archive=run.PUBLIC/'metric_groups.zip'
    if not archive.exists():pytest.skip('Complete scalar bundle not ready')
    forbidden={'features','targets','rollouts','floor_rollout','history_x','history_y',
        'current_x','current_y','future_endpoint','model','optimizer'}
    def inspect(v):
        if isinstance(v,dict):
            assert not set(v)&forbidden
            for item in v.values():inspect(item)
        elif isinstance(v,list):
            for item in v:inspect(item)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==72
        for name in z.namelist():
            assert name.startswith('groups/') and name.endswith('.json')
            raw=z.read(name);assert b'/Users/' not in raw
            inspect(json.loads(raw,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v))))
