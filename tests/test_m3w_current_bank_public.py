import builtins
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import pytest
from scripts import report_m3w_current_bank_deferral as report
from tests.test_m3w_active_bank_readout import fixture

run=report.run


def test_supplementary_usage_includes_unknown_rows_and_fallback():
    p=fixture('s0',missing=True)[0]['new_neural']
    choices={n:np.full(2,6) for n in run.evaluation.NAMES if n!='floor'}
    result=run.evaluation.readout(p,choices,{n:np.ones(2) for n in choices},1.5,np.zeros((2,2)))
    value=report.auxiliary([dict(result=result)])
    assert value['tdef_raw']['repeated_row_views']==2 and value['tdef_raw']['switch_rate']==1.
    assert value['floor']['action_counts']['-1']==2 and value['floor']['switch_rate']==0.


def test_figure_builds_with_absent_private_and_public_directories(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'PUBLIC',tmp_path/'public');monkeypatch.setattr(run,'PRIVATE',tmp_path/'private')
    interval=dict(point=1.,CI95=[.2,1.5])
    summary=dict(accuracy={n:interval for n in run.evaluation.NAMES},
        effects=dict(guarded_tdef_minus_ade=dict(full={k:interval for k in ('known_pp','lower_pp')})))
    report.figure(dict(summary=summary))
    assert (run.PUBLIC/'figures/current_bank_deferral.svg').exists() and (run.PRIVATE/'current_bank_deferral.png').exists()


def test_actual_public_report_avoids_private_inputs_when_complete(monkeypatch):
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Real complete experiment not yet verified')
    po,fo,ioo=Path.open,builtins.open,io.open
    def guard(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).absolute()
            if any(p.is_relative_to(run.ROOT/n) for n in ('data','external_data')):raise AssertionError('Private input read')
    def path_open(p,*a,**kw):guard(p);return po(p,*a,**kw)
    def file_open(p,*a,**kw):guard(p);return fo(p,*a,**kw)
    def io_open(p,*a,**kw):guard(p);return ioo(p,*a,**kw)
    monkeypatch.setattr(Path,'open',path_open);monkeypatch.setattr(builtins,'open',file_open);monkeypatch.setattr(io,'open',io_open)
    value=report.build()
    assert value['summary']['groups']==72 and value['training']['freeze']['primary_optimizer_updates']==208320


def test_complete_bundle_contains_scalars_not_raw_arrays():
    path=run.PUBLIC/'metric_groups.zip'
    if not path.exists():pytest.skip('Full scalar bundle not yet complete')
    forbidden={'features','targets','rollouts','history_x','history_y','future_endpoint','model','optimizer'}
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
            raw=z.read(name);assert b'/Users/' not in raw;inspect(json.loads(raw))
