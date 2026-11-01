import builtins
import io
import json
import os
from pathlib import Path
import zipfile
import pytest
from scripts import report_m3w_reference_support_preflight as reference
from scripts import report_m3w_query_easy_support as query


@pytest.mark.parametrize('report',(reference,query))
def test_frozen_definitions_and_no_role_promotion(report):
    run=report.run;reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():assert run.sha(run.ROOT/rel)==digest
    assert not any(reg[k] for k in ('new_training','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))


@pytest.mark.parametrize('report',(reference,query))
def test_complete_original_cohort_if_ready(report):
    if not (report.run.PUBLIC/'verification.json').exists():pytest.skip('Full exact replay not complete')
    v=report.build();s=v['summary']
    assert s['groups']==72 and s['unique_target_rows']==149510 and s['unique_recordings']==74
    assert not s['new_training'] and not s['deployment_changed'] and not s['submission_ready']


@pytest.mark.parametrize('report',(reference,query))
def test_public_reporting_has_no_private_input_reads(report,monkeypatch):
    if not (report.run.PUBLIC/'verification.json').exists():pytest.skip('Full public evidence not ready')
    def checked(fn):
        def call(path,*a,**kw):
            if isinstance(path,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(path)).absolute()
                if any(p.is_relative_to(report.run.ROOT/k) for k in ('data','external_data')):raise AssertionError('No private report inputs')
            return fn(path,*a,**kw)
        return call
    monkeypatch.setattr(Path,'open',checked(Path.open));monkeypatch.setattr(builtins,'open',checked(builtins.open));monkeypatch.setattr(io,'open',checked(io.open))
    assert report.build()['summary']['groups']==72


@pytest.mark.parametrize('report',(reference,query))
def test_scalar_metric_bundle_contains_no_arrays_or_weights(report):
    p=report.run.PUBLIC/'metric_groups.zip'
    if not p.exists():pytest.skip('Full metric bundle not ready')
    forbidden={'features','targets','rollouts','history_x','history_y','current_x','current_y','future_endpoint','model','optimizer'}
    def inspect(v):
        if isinstance(v,dict):
            assert not forbidden&set(v)
            for x in v.values():inspect(x)
        elif isinstance(v,list):
            for x in v:inspect(x)
    with zipfile.ZipFile(p) as z:
        assert len(z.namelist())==72
        for n in z.namelist():
            assert n.startswith('groups/') and n.endswith('.json')
            raw=z.read(n);assert b'/Users/' not in raw;inspect(json.loads(raw))
