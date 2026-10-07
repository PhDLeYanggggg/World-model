import ast
import subprocess
from types import SimpleNamespace

import pytest

from scripts import allow_m3w_portability_backfill as run


def pending():
    return dict(JobId='37837636',JobName='m3w_easy_node_portability',
        JobState='PENDING',ReqNodeList='erc-hpc-comp188',NumCPUs='4',
        MinMemoryNode='8G',TimeLimit='01:00:00',TimeMin='N/A',
        RunTime='00:00:00',Restarts='0')


def test_exact_pending_request_required():
    run.check_pending(pending())


@pytest.mark.parametrize('key,value',[
    ('JobId','another_job'),('JobName','another_name'),('JobState','RUNNING'),
    ('JobState','COMPLETED'),('TimeMin','00:10:00'),('TimeLimit','00:10:00'),
    ('NumCPUs','2'),('MinMemoryNode','16G'),('ReqNodeList','other_node'),
    ('RunTime','00:00:01'),('Restarts','1')])
def test_changed_request_refused_before_mutation(key,value):
    fields=pending();fields[key]=value
    with pytest.raises(ValueError,match=key):
        run.check_pending(fields)


def test_observation_timeout_is_not_success_or_retry(monkeypatch):
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(subprocess,'run',fake)
    result=run.update()
    assert result['outcome']=='unknown_inspect_do_not_retry'
    assert result['returncode'] is None
    assert calls==[['scontrol','update','JobId=37837636','TimeMin=00:10:00']]


def test_scheduler_rejection_retained(monkeypatch):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(
        returncode=1,stdout='',stderr='Refused'))
    result=run.update()
    assert result['returncode']==1 and result['stderr']=='Refused'


def test_intent_and_integrity_checks_precede_single_update():
    ast.parse(run.REMOTE)
    assert run.REMOTE.index('check_pending(before)')<run.REMOTE.index('once(intent,')<run.REMOTE.index('result=update()')
    assert 'assert not intent.exists() and not receipt.exists()' in run.REMOTE
    assert "'sbatch'" not in run.REMOTE and "'scancel'" not in run.REMOTE
    assert 'scientific_contract_changed=False' in run.REMOTE
