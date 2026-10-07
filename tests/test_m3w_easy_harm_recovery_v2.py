import ast
import copy
import inspect

import pytest

from scripts import train_m3w_easy_harm_recovery_v2 as run
from scripts import manage_m3w_easy_harm_recovery_v2 as manager
from tests.test_m3w_easy_harm_verified import state
from src.world_model import m3w_easy_harm_deviance_training as api


def reference():
    return dict(identity={'case':1},preserve_v1=False,reference_checkpoint={'path':'old','sha256':'1'*64})


def test_exact_replay_not_generic_tolerance():
    old=state('none'); old['model']['w'][0]+=.2
    result=run.compare_reference(state('quadratic'),old,reference(),state('none'),api)
    assert result['historical_control_exact'] is False
    assert result['original_implementation_control_exact'] is True
    bad=state('none');bad['model']['w'][0]+=1e-6
    with pytest.raises(AssertionError):run.compare_reference(state('quadratic'),old,reference(),bad,api)
    with pytest.raises(AssertionError):run.compare_reference(state('quadratic'),old,None,None,api)


@pytest.mark.parametrize('key',['identity','step','initial_model','input_hashes','preprocess','draw_hash','sampler_rng'])
def test_metadata_and_identity_cannot_be_waived(key):
    old=state('none');old['model']['w'][0]+=.2;old[key]={'wrong':1}
    with pytest.raises((AssertionError,ValueError,TypeError)):
        run.compare_reference(state('quadratic'),old,reference(),state('none'),api)


def test_registered_mismatch_cannot_silently_become_historical():
    with pytest.raises(ValueError):run.compare_reference(state('quadratic'),state('none'),reference(),state('none'),api)


def test_fit_requires_exact_reference_and_correct_amendment():
    ref=reference();reg={'references':{'known':ref}}
    row=dict(identity=ref['identity'],arm='quadratic',step=2000,validation_scored=False,
             independent_roles_read=False,new_forecaster=False,parent_control_exact=False,
             historical_control_exact=False,original_implementation_control_exact=True,
             floating_tolerance_relaxed=False,reference_checkpoint=ref['reference_checkpoint'],amendment_sha256='v2')
    assert run.validate_fit(row,'known',reg,'v1','v2')=='replay'
    for key,value in [('validation_scored',True),('parent_control_exact',True),('step',1999),
                      ('amendment_sha256','v1'),('floating_tolerance_relaxed',True)]:
        changed={**row,key:value}
        with pytest.raises(ValueError):run.validate_fit(changed,'known',reg,'v1','v2')
    with pytest.raises(ValueError):run.validate_fit(row,'unknown',reg,'v1','v2')
    ref['preserve_v1']=True
    assert run.validate_fit({**row,'amendment_sha256':'v1'},'known',reg,'v1','v2')=='replay'


def test_only_missing_shard_submitted_no_login_numerics():
    ast.parse(manager.SUBMIT)
    assert '--array=0' in manager.SUBMIT and '--array=0,' not in manager.SUBMIT
    assert 'import torch' not in manager.SUBMIT and 'import numpy' not in manager.SUBMIT
    assert 'execution_archive_v2' in manager.SUBMIT
    assert "['scancel','37815222']" in manager.SUBMIT
    assert 'unknown_timeout_inspect_do_not_resubmit' in manager.SUBMIT
    assert 'api.fit' not in inspect.getsource(run.train)


def test_registry_closure_remains_exact18_and110():
    from pathlib import Path
    import json
    observed=json.loads((manager.HOME/'partial_training_observation_20261006T1256Z.json').read_text())
    assert len(observed['fits'])==110 and len(observed['missing'])==34
    complete=json.loads((manager.HOME/'control_diagnostic_v2/complete.json').read_text())
    assert len(complete['results'])==17
    for ref in complete['results']:
        path=manager.ROOT/ref['path']
        assert manager.manager.base.sha(path)==ref['sha256']
        assert json.loads(path.read_text())['comparisons']['old_direct_vs_new_direct']['all_control_fields_exact']
