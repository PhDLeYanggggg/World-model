import builtins
import json
import os
from pathlib import Path
import zipfile
import pytest
from scripts import report_m3w_decision_cost_critic as report

run=report.run


def test_registered_definition_is_unchanged():
    reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items(): assert run.sha(run.ROOT/rel)==digest
    assert not reg['independent_roles_opened'] and not reg['inference_outcomes_used']
    assert reg['primary_optimizer_updates']==36000 and reg['cached_verified_regression_controls']==18


def test_complete_real_fits_and_readout_controls_if_available():
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Complete original scoring verification not ready')
    v=report.build();s=v['summary']
    assert s['groups']==72 and s['unique_target_rows']==149510 and s['unique_recordings']==74
    assert v['fit_stats']['complete_fits']==18 and v['full_refit']['scientific_state_exact']
    assert v['replay']['all72_frozen_actions_inference_and_scores_exact']
    assert not v['replay']['optimization_replayed'] and not s['independent_confirmation']
    assert not s['deployment_changed'] and not s['submission_ready']


def test_public_report_requires_no_private_data(monkeypatch):
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Complete public evidence not ready')
    old_path=Path.open;old_open=builtins.open
    def guard(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).absolute()
            if p.is_relative_to(run.ROOT/'data') or p.is_relative_to(run.ROOT/'external_data'):
                raise AssertionError('Public aggregate report must not read private data')
    def path_open(path,*a,**kw):guard(path);return old_path(path,*a,**kw)
    def file_open(path,*a,**kw):guard(path);return old_open(path,*a,**kw)
    monkeypatch.setattr(Path,'open',path_open);monkeypatch.setattr(builtins,'open',file_open)
    assert report.build()['summary']['groups']==72


def test_metric_bundle_contains_only_scalar_aggregates():
    archive=run.PUBLIC/'metric_groups.zip'
    if not archive.exists():pytest.skip('Complete metric bundle not ready')
    forbidden={'features','targets','rollouts','floor_rollout','history_x','history_y','current_x','current_y','future_endpoint','model','optimizer'}
    def inspect(v):
        if isinstance(v,dict):
            assert not set(v)&forbidden
            for value in v.values():inspect(value)
        elif isinstance(v,list):
            for value in v:inspect(value)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==72
        for name in z.namelist():
            assert name.startswith('groups/') and name.endswith('.json')
            raw=z.read(name);assert b'/Users/' not in raw
            inspect(json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value))))
