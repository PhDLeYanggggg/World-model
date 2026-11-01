import builtins
import io
import json
import os
from pathlib import Path
import zipfile

import pytest
from scripts import report_m3w_coherent_cost_projection as report

run=report.run


def test_registered_definition_and_original_model_freeze_unchanged():
    reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():assert run.sha(run.ROOT/rel)==digest
    for name,digest in reg['parent_files'].items():assert run.sha(run.parent.PUBLIC/name)==digest
    assert reg['frozen_cost_models']==36 and not reg['new_training']
    assert not reg['independent_roles_opened'] and not reg['deployment_changed']


def test_complete_original_roster_and_every_control_if_available():
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Complete verification not ready')
    v=report.build();s=v['summary']
    assert s['groups']==72 and s['unique_target_rows']==149510 and s['unique_recordings']==74
    assert not s['new_training'] and not s['whole_pipeline_OOF'] and not s['independent_confirmation']
    assert not v['replay']['optimization_replayed']
    assert not s['deployment_changed'] and not s['submission_ready']
    assert s['fitting_gates']['projected_moments_coherent']


def test_public_builder_cannot_read_private_inputs(monkeypatch):
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Complete public readout not ready')
    po,fo,ioo=Path.open,builtins.open,io.open
    def guard(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).absolute()
            if any(p.is_relative_to(run.ROOT/name) for name in ('data','external_data')):
                raise AssertionError('Only public scalar evidence belongs in report rebuild')
    def path_open(path,*a,**kw):guard(path);return po(path,*a,**kw)
    def file_open(path,*a,**kw):guard(path);return fo(path,*a,**kw)
    def io_open(path,*a,**kw):guard(path);return ioo(path,*a,**kw)
    monkeypatch.setattr(Path,'open',path_open);monkeypatch.setattr(builtins,'open',file_open);monkeypatch.setattr(io,'open',io_open)
    assert report.build()['summary']['groups']==72


def test_bundle_contains_only72public_scalar_groups():
    path=run.PUBLIC/'metric_groups.zip'
    if not path.exists():pytest.skip('Complete metric archive not ready')
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


def test_empty_pair_or_selected_support_is_not_imputed_as_observed_zero():
    def row(n,value):
        return dict(result=dict(recordings={},record_tails={},pair_proxy=dict(pair_requested_step_views=n,
            mean_positive_distance_contraction_over_frozen_scale={a:value for a in run.evaluation.NAMES}),
            pointwise_diagnostic={a:dict(supports={}) for a in run.api.ARMS},allocation=dict(queries=1,solver_calls=0,nonoptimal={},reason_counts={}),
            selected_cost_diagnostic={a:dict(known_selected_rows=0,normalized_bias=None,
                selected_harm_underprediction_rows=0,selected_easy_harm_underprediction_rows=0)
                for a in ('raw_locked_joint','raw_learned_joint','locked_joint','learned_joint')}))
    empty=row(0,None);v=report.supplementary([empty])
    assert v['pair_and_tail']['learned_joint']['mean_pair_contraction'] is None
    assert v['selected_cost_diagnostic']['learned_joint']['pooled_normalized_bias'] is None
    assert report.supplementary([empty,row(4,2.)])['pair_and_tail']['learned_joint']['mean_pair_contraction']==2.
    with pytest.raises(ValueError):report.supplementary([row(4,None)])


def test_figure_creates_absent_private_output_directory(tmp_path,monkeypatch):
    public=tmp_path/'public';public.mkdir();private=tmp_path/'absent_output'
    monkeypatch.setattr(run,'PUBLIC',public);monkeypatch.setattr(run,'PRIVATE',private)
    interval=dict(point=1.,CI95=[.5,1.5])
    summary=dict(accuracy={a:interval for a in ('raw_locked_joint','locked_joint','raw_learned_joint','learned_joint')},
        effects=dict(learned_projection=dict(full={k:interval for k in ('known_pp','lower_pp')})))
    report.figure(dict(summary=summary))
    assert (private/'coherent_cost_readout.png').is_file()
    assert (public/'figures/coherent_cost_readout.svg').is_file()
