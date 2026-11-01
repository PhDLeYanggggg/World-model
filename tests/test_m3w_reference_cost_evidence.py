import builtins
import copy
import io
import json
import os
from pathlib import Path
import zipfile

import pytest
from scripts import report_m3w_reference_cost_refit as report

run=report.run


def fits():
    p=run.PUBLIC/'training_freeze.json'
    if not p.exists():pytest.skip('Full36training freeze not ready')
    return [json.loads((run.ROOT/ref['path']).read_text()) for ref in json.loads(p.read_text())['fits']]


def test_all_registered_scientific_definitions_remain_frozen():
    reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():assert run.sha(run.ROOT/rel)==digest
    assert reg['new_cost_fits']==36 and reg['warm_start_not_from_scratch']
    assert reg['reference_trainable_capacity_difference']==262
    assert not reg['independent_roles_opened'] and not reg['deployment_changed']


def test_complete_real_cost_training_and_full_refits():
    values=fits();stats=report.training_statistics(values)
    freeze=json.loads((run.PUBLIC/'training_freeze.json').read_text())
    assert stats['primary_optimizer_updates']==freeze['primary_optimizer_updates']
    assert stats['complete_fits']==36 and not stats['active_capacity_matched']
    assert min(t['minimum_query_visits'] for t in stats['arms'].values())>=4
    assert stats['arms']['locked_references']['trainable_parameters']==30857
    assert stats['arms']['learned_references']['trainable_parameters']==31119


@pytest.mark.parametrize('problem',('missing','duplicate','budget','coverage','unknown','draws','features','active'))
def test_incomplete_or_unmatched_cost_training_is_rejected(problem):
    values=copy.deepcopy(fits())
    if problem=='missing':values.pop()
    elif problem=='duplicate':values[-1]=values[0]
    elif problem=='budget':values[0]['step']=100
    elif problem=='coverage':values[0]['minimum_query_visits']=3
    elif problem=='unknown':values[0]['unknown_rows_with_supervised_labels']=1
    elif problem=='draws':values[0]['draw_hash']='changed'
    elif problem=='features':values[0]['input_hashes']['features']='changed'
    elif problem=='active':values[0]['trainable_parameters']=1
    with pytest.raises(ValueError):report.training_statistics(values)


def test_complete_policy_readout_and_original_controls():
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Full policy verification not ready')
    v=report.build();s=v['summary']
    assert s['groups']==72 and s['unique_target_rows']==149510 and s['unique_recordings']==74
    assert len(s['fitting_gates'])==8 and not s['whole_pipeline_OOF']
    assert not s['independent_confirmation'] and not s['deployment_changed'] and not s['submission_ready']
    assert not v['replay']['optimization_replayed']


def test_public_rebuild_without_private_input_reads(monkeypatch):
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Full public evidence not ready')
    po,fo,ioo=Path.open,builtins.open,io.open
    def guard(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            value=Path(os.fsdecode(path)).absolute()
            if any(value.is_relative_to(run.ROOT/name) for name in ('data','external_data')):
                raise AssertionError('Public reports cannot read private inputs')
    def path_open(path,*a,**kw):guard(path);return po(path,*a,**kw)
    def file_open(path,*a,**kw):guard(path);return fo(path,*a,**kw)
    def io_open(path,*a,**kw):guard(path);return ioo(path,*a,**kw)
    monkeypatch.setattr(Path,'open',path_open);monkeypatch.setattr(builtins,'open',file_open);monkeypatch.setattr(io,'open',io_open)
    assert report.build()['summary']['groups']==72


def test_scalar_only_full_metric_archive():
    path=run.PUBLIC/'metric_groups.zip'
    if not path.exists():pytest.skip('Scalar archive not ready')
    forbidden={'features','targets','rollouts','floor_rollout','history_x','history_y',
        'current_x','current_y','future_endpoint','model','optimizer'}
    def check(v):
        if isinstance(v,dict):
            assert not set(v)&forbidden
            for x in v.values():check(x)
        elif isinstance(v,list):
            for x in v:check(x)
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==72
        for name in z.namelist():
            assert name.startswith('groups/') and name.endswith('.json')
            raw=z.read(name);assert b'/Users/' not in raw
            check(json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))))


def test_no_pair_support_remains_undefined_not_observed_zero():
    def row(n,value):
        return dict(result=dict(pair_proxy=dict(pair_requested_step_views=n,
            mean_positive_distance_contraction_over_frozen_scale={a:value for a in run.evaluation.NAMES}),
            recordings={},record_tails={},allocation=dict(queries=1,solver_calls=0),
            pointwise_diagnostic={a:dict(supports={}) for a in run.api.ARMS}))
    empty=row(0,None);positive=row(3,2.)
    assert report.supplementary([empty])['pair_and_tail']['learned_joint']['pair_contraction_proxy'] is None
    assert report.supplementary([empty,positive])['pair_and_tail']['learned_joint']['pair_contraction_proxy']==2.
    with pytest.raises(ValueError):report.supplementary([row(3,None)])


def test_unobserved_target_values_cannot_change_supervised_loss_or_gradients():
    from tests.test_m3w_reference_cost_refit import warm
    from tests.test_m3w_coupled_benefit_risk import data
    api=run.api;d=data();changed=copy.deepcopy(d)
    assert (~d['known']).any()
    changed['targets'][~d['known']]=1e12
    ix=api.np.arange(8);seg=api.np.repeat(api.np.arange(4),2)
    args=api.parent.batch(d['causal'],ix,seg,4)
    normalizer=api.base.units.query_normalizer(d)
    for arm in api.ARMS:
        model=api.initialize(warm(d),arm);results=[]
        for source in (d,changed):
            model.zero_grad(set_to_none=True)
            terms=api.losses(model(*args),source,ix,args[-2],4,normalizer)
            terms['total'].backward()
            results.append(dict(loss=terms['total'].detach().clone(),
                gradients={k:p.grad.clone() if p.grad is not None else None for k,p in model.named_parameters()}))
        api.parent.exact(*results)
