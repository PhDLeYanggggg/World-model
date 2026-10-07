import ast
import subprocess
from types import SimpleNamespace

import pytest

from scripts import observe_m3w_recovery_resilient as run


def test_timeout_does_not_become_failed_or_completed_job(monkeypatch):
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(subprocess,'run',fake)
    result=run.query(['squeue','-j','37837636'])
    assert result['observation']=='timeout_job_state_unknown'
    assert result['returncode'] is None and result['stdout'] is None
    assert len(calls)==1


def test_independent_accounting_still_returned_after_queue_timeout(monkeypatch):
    def fake(command,**kwargs):
        if command[0]=='squeue':
            raise subprocess.TimeoutExpired(command,kwargs['timeout'])
        return SimpleNamespace(returncode=0,stdout='37837636|PENDING|00:00:00|0:0\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    queue=run.query(['squeue','-j','37837636'])
    accounting=run.query(['sacct','-j','37837636'])
    assert queue['returncode'] is None
    assert accounting['returncode']==0 and '|PENDING|' in accounting['stdout']


def test_missing_command_is_observation_failure(monkeypatch):
    def fake(*args,**kwargs):
        raise FileNotFoundError('missing scheduler client')
    monkeypatch.setattr(subprocess,'run',fake)
    result=run.query(['squeue','-j','37835856'])
    assert result['observation']=='query_unavailable_job_state_unknown'


@pytest.mark.parametrize('command',[
    ['scancel','-j','37837636'],['sbatch','script'],['squeue','-j','foreign'],
    ['sacct','-j','37837636,foreign'],['squeue','--me']])
def test_other_jobs_or_mutations_are_not_allowed(command):
    with pytest.raises(ValueError):
        run.query(command)


def test_observer_retains_artifact_and_stale_heartbeat_boundary():
    ast.parse(run.REMOTE)
    assert 'remote_modified=False' in run.REMOTE
    assert 'jobs_resubmitted=False' in run.REMOTE
    assert 'file_is_not_live_process_evidence=True' in run.REMOTE
    assert 'observed_file_only_not_scheduler_confirmation=True' in run.REMOTE
    assert 'import torch' not in run.REMOTE and 'import numpy' not in run.REMOTE
