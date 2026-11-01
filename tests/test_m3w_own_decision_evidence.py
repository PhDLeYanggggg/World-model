import json
from pathlib import Path
import numpy as np
import pytest
from scripts import run_m3w_own_decision_cost as run
from tests.test_m3w_coupled_benefit_risk import data


def test_allocator_adapter_changes_names_not_native_decisions():
    d=data();c=d['causal'];p=run.previous.projection.project(c['prediction'],c['allowed'])
    predicted={a:p.copy() for a in run.api.ARMS};anchor=np.full(len(p),-1,np.int64)
    args=(c['allowed'],c['recordings'],c['frames'],c['ids'],anchor)
    actual=run.api.allocate(predicted,*args,seconds=2.)
    old=run.api.allocation.allocate(dict(zip(run.api.allocation.ARMS,predicted.values())),*args,seconds=2.)
    for new,oldname in zip(('fixed','own'),('locked','learned')):
        for mode in ('independent','joint','matched'):
            np.testing.assert_array_equal(actual[0][new+'_'+mode],old[0][oldname+'_'+mode])
            np.testing.assert_array_equal(actual[1][new+'_'+mode],old[1][oldname+'_'+mode])
    assert actual[2]['solver_calls']==old[2]['solver_calls'] and not actual[2]['labels_used']
    np.testing.assert_array_equal(actual[0]['frozen_projected_joint'],anchor)


def test_allocator_real_resume_keeps_same_endpoint_and_every_query_count():
    d=data();c=d['causal'];p=run.previous.projection.project(c['prediction'],c['allowed'])
    args=({a:p.copy() for a in run.api.ARMS},c['allowed'],c['recordings'],c['frames'],c['ids'],np.full(len(p),-1,np.int64))
    states=[]
    class Stop(Exception):pass
    def stop(v):states.append(v);raise Stop()
    with pytest.raises(Stop):run.api.allocate(*args,seconds=2.,checkpoint=stop)
    direct=run.api.allocate(*args,seconds=2.);recovered=run.api.allocate(*args,seconds=2.,resume_state=states[0])
    run.api.parent.exact(direct,recovered)


def test_registered_definition_and_original_roles_when_registered():
    if not run.REG.exists():pytest.skip('Definition not yet registered')
    reg=json.loads(run.REG.read_text())
    for path,digest in dict(reg['bindings'],**reg['dependencies']).items():assert run.sha(run.ROOT/path)==digest
    for path,digest in reg['parent_files'].items():assert run.sha(run.previous.PUBLIC/path)==digest
    assert reg['new_cost_fits']==36 and reg['total_and_trainable_parameters']==31119
    assert not reg['new_trajectory_training'] and not reg['independent_roles_opened']


def test_real_fits_and_original_roster_if_complete():
    if not (run.PUBLIC/'verification.json').exists():pytest.skip('Complete real result not available')
    trained=json.loads((run.PUBLIC/'training_freeze.json').read_text());s=json.loads((run.PUBLIC/'summary.json').read_text())
    assert len(trained['fits'])==36 and trained['primary_optimizer_updates']==208320 and trained['exact_full_refit_updates']==14336
    assert s['groups']==72 and s['unique_target_rows']==149510 and s['unique_recordings']==74
    assert not s['whole_pipeline_OOF'] and not s['independent_confirmation'] and not s['deployment_changed']
    for ref in trained['fits']:
        assert run.sha(run.ROOT/ref['path'])==ref['sha256']
        x=json.loads((run.ROOT/ref['path']).read_text())
        assert x['minimum_query_visits']>=4 and x['never_drawn_queries']==0
        assert x['parameters']==x['trainable_parameters']==31119 and x['unknown_rows_with_supervised_labels']==0
