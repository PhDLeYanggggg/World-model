import builtins
import io
import json
import os
from pathlib import Path
import zipfile
import pytest
from scripts import report_m3w_frozen_cost_support as report

run=report.run


def test_diagnostic_registration_preserves_roles_and_frozen_parent():
    reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():assert run.sha(run.ROOT/rel)==digest
    assert reg['post_hoc_parent_outcomes_already_read'] and reg['future_labels_used_by_diagnostic_oracle']
    assert not any(reg[k] for k in ('new_training','held_scored','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))
    assert reg['parent_summary_sha256']==run.sha(run.parent.PUBLIC/'summary.json')


def test_complete72_diagnosis_if_ready():
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Full diagnostic/replay not complete')
    v=report.build();s=v['summary']
    assert s['groups']==72 and s['unique_target_rows']==149510 and s['unique_recordings']==74
    assert s['pointwise_query_budget_witnesses_exact'] and s['oracle_not_a_model_result']
    assert not s['independent_confirmation'] and not s['new_training']
    for a in run.api.SUPPORTS:assert s['supports'][a]['required_record_views']==444


def test_public_scalar_report_requires_no_private_input(monkeypatch):
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Full public evidence not ready')
    old_path=Path.open;old_open=builtins.open;old_io=io.open
    def guard(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).absolute()
            if any(p.is_relative_to(run.ROOT/k) for k in ('data','external_data')):
                raise AssertionError('No private inputs in public support report')
    def path_open(path,*a,**kw):guard(path);return old_path(path,*a,**kw)
    def file_open(path,*a,**kw):guard(path);return old_open(path,*a,**kw)
    def io_open(path,*a,**kw):guard(path);return old_io(path,*a,**kw)
    monkeypatch.setattr(Path,'open',path_open);monkeypatch.setattr(builtins,'open',file_open);monkeypatch.setattr(io,'open',io_open)
    assert report.build()['summary']['groups']==72


def test_support_bundle_is_scalar_only_if_ready():
    archive=run.PUBLIC/'metric_groups.zip'
    if not archive.exists():pytest.skip('Full public bundle not ready')
    forbidden={'features','targets','rollouts','history_x','history_y','current_x','current_y','future_endpoint','model','optimizer'}
    def inspect(v):
        if isinstance(v,dict):
            assert not forbidden&set(v)
            for value in v.values():inspect(value)
        elif isinstance(v,list):
            for value in v:inspect(value)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==72
        for name in z.namelist():
            assert name.startswith('groups/') and name.endswith('.json')
            raw=z.read(name);assert b'/Users/' not in raw
            inspect(json.loads(raw))


def test_zero_reference_excess_is_not_a_defined_row_risk_ratio():
    from tests.test_m3w_frozen_cost_support import packet
    p,prediction,controls=packet();p['y'][0]=0.;p['y'][0,0,1]=2.
    prediction[0,:,0]=2.;prediction[0,:,1]=0.
    r=run.api.diagnose(p,prediction,controls)
    assert r['calibration']['selected_joint']['predicted_pointwise_safe_but_observed_row_risk_over2_percent']==1
    assert r['calibration']['selected_joint']['row_diagnostic_not_recording_safety']
