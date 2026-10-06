import copy
import ast
import inspect

import pytest
import torch

from scripts import train_m3w_easy_harm_verified as run
from scripts import resume_m3w_easy_harm_verified as manager
from src.world_model import m3w_easy_harm_deviance_training as api


def state(arm):
    row = {k:1 for k in ('initial_model','input_hashes','preprocess','settings','seed',
                         'sampler_rng','torch_rng','draw_hash','row_draws','queries')}
    row.update(identity={'case':1},arm=arm,step=2000,
               model={'w':torch.tensor([1.,2.])},optimizer={'w':torch.tensor([0.,1.])})
    return row


def test_historical_exact_has_no_exception_or_replay():
    result=run.check_reference(state('quadratic'),state('none'),None,{'different':1},api)
    assert result['historical_control_exact'] and result['original_implementation_control_exact']


def test_registered_exception_requires_exact_original_replay_not_tolerance():
    new=state('quadratic'); old=state('none'); old['model']['w'][0]+=1e-6
    result=run.check_reference(new,old,state('none'),{'case':1},api)
    assert not result['historical_control_exact']
    assert result['original_implementation_control_exact'] and not result['floating_tolerance_relaxed']
    bad=state('none');bad['model']['w'][0]+=1e-7
    with pytest.raises(AssertionError): run.check_reference(new,old,bad,{'case':1},api)


@pytest.mark.parametrize('changed',['identity','preprocess','input_hashes','initial_model','step','draw_hash'])
def test_no_general_mismatch_waiver(changed):
    new=state('quadratic');old=state('none');old['model']['w'][0]+=1e-6
    if changed=='identity': old['identity']={'case':2}
    else: old[changed]=2
    with pytest.raises(AssertionError): run.check_reference(new,old,state('none'),{'case':1},api)


def test_missing_reference_cannot_pass():
    new=state('quadratic');old=state('none');old['model']['w'][0]+=1e-6
    with pytest.raises(ValueError): run.check_reference(new,old,None,{'case':1},api)


def test_scheduler_wrapper_has_preservation_and_no_numeric_login_work():
    for source in (manager.INVENTORY,manager.SUBMIT):
        ast.parse(source)
        assert 'import torch' not in source and 'import numpy' not in source
    assert '--array=0,2,3%3' in manager.SUBMIT
    assert 'execution_archive_v1' in manager.SUBMIT
    assert '37814169_1' in manager.SUBMIT
    assert 'old_pending_cancelled' in manager.SUBMIT
    assert 'api.fit' not in inspect.getsource(run.train)
