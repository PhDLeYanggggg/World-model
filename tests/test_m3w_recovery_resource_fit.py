import ast
import subprocess

import pytest

from scripts import fit_m3w_recovery_resources as run


def records():
    return '\n'.join(job+'|COMPLETED|0:0|00:05:00|1613184K' for job in
        ('37814169_1.batch','37815216_2.batch','37815216_3.batch','37817976.batch'))


def pending():
    return dict(JobId='37835856',JobName='m3w_easy_recovery_v2_train',
        JobState='PENDING',ReqNodeList='erc-hpc-comp186',NumCPUs='4',
        MinMemoryNode='16G',TimeLimit='12:00:00',TimeMin='01:00:00',
        RunTime='00:00:00',Restarts='0')


def test_complete_low_memory_evidence_and_pending_job():
    assert len(run.memory_evidence(records()))==4
    run.check_pending(pending())


@pytest.mark.parametrize('change',['missing','duplicate','failed','zero','large','unknown'])
def test_refuse_unjustified_memory_reduction(change):
    text=records()
    if change=='missing':text='\n'.join(text.splitlines()[:-1])
    elif change=='duplicate':text+='\n'+text.splitlines()[0]
    elif change=='failed':text=text.replace('COMPLETED','FAILED',1)
    elif change=='zero':text=text.replace('1613184K','0K',1)
    elif change=='large':text=text.replace('1613184K','3G',1)
    else:text=text.replace('1613184K','',1)
    with pytest.raises(ValueError):run.memory_evidence(text)


@pytest.mark.parametrize('key,value',[
    ('JobId','37837636'),('JobState','RUNNING'),('JobState','COMPLETED'),
    ('MinMemoryNode','8G'),('TimeMin','00:10:00'),('NumCPUs','2'),
    ('ReqNodeList','erc-hpc-comp188'),('RunTime','00:00:01'),('Restarts','1')])
def test_no_mutation_after_job_or_request_changes(key,value):
    fields=pending();fields[key]=value
    with pytest.raises(ValueError,match=key):run.check_pending(fields)


def test_single_update_timeout_preserved_as_unknown(monkeypatch):
    calls=[]
    def fake(command,**kwargs):
        calls.append(command);raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(subprocess,'run',fake)
    out=run.update()
    assert out['returncode'] is None and out['outcome']=='unknown_inspect_do_not_retry'
    assert calls==[['scontrol','update','JobId=37835856','MinMemoryNode=8192','TimeMin=00:10:00']]


def rejection():
    return dict(outcome='command_returned',returncode=1,
                stderr='scontrol: error: Invalid MinMemoryNode value: 8G\n',
                before=pending(),after=pending())


def test_confirmed_no_change_rejection_allows_syntax_correction():
    run.require_rejected_prior(rejection())


@pytest.mark.parametrize('change',['unknown','success','other_error','changed','missing_after'])
def test_no_retry_after_ambiguous_or_applied_update(change):
    prior=rejection()
    if change=='unknown':prior['outcome']='unknown_inspect_do_not_retry';prior['returncode']=None
    elif change=='success':prior['returncode']=0
    elif change=='other_error':prior['stderr']='Permission denied'
    elif change=='changed':prior['after']['MinMemoryNode']='8G'
    else:prior['after']=None
    with pytest.raises(ValueError):run.require_rejected_prior(prior)


def test_remote_only_changes_reservation_after_registered_checks():
    ast.parse(run.REMOTE)
    assert run.REMOTE.index('verify(root)')<run.REMOTE.index('once(intent,')<run.REMOTE.index('result=update()')
    assert "'sbatch'" not in run.REMOTE and "'scancel'" not in run.REMOTE
    assert 'node_changed=False' in run.REMOTE and 'scientific_contract_changed=False' in run.REMOTE
