import subprocess
from types import SimpleNamespace

import pytest

from scripts import collect_m3w_create_web_accounting as worker


@pytest.mark.parametrize('status,code,expected', [
    ('COMPLETED', '0:0', True), ('FAILED', '1:0', False),
    ('COMPLETED', '1:0', False), ('PENDING', '0:0', False),
    ('RUNNING', '0:0', False), ('CANCELLED', '0:0', False)])
def test_exact_successful_terminal_accounting(status, code, expected):
    row = dict(returncode=0, stdout=f'37835859|{status}|00:00:06|{code}|\n')
    assert worker.successful_accounting(row, '37835859') is expected


def test_wrong_job_or_timeout_is_not_success():
    row = dict(returncode=0, stdout='37835859.batch|COMPLETED|00:00:06|0:0|\n')
    assert not worker.successful_accounting(row, '37835859')
    row = dict(returncode=None, stdout='')
    assert not worker.successful_accounting(row, '37835859')


@pytest.mark.parametrize('kind,job', [('sbatch','37835859'), ('scancel','37835859'),
                                    ('sacct','unrelated'), ('squeue','37835859,other')])
def test_other_jobs_and_mutations_rejected(kind, job):
    with pytest.raises(ValueError):
        worker.query(kind, job)


def test_owned_query_uses_explicit_cluster_date_and_bounded_wait(monkeypatch):
    calls = []
    def execute(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout='record', stderr='')
    monkeypatch.setattr(subprocess, 'run', execute)
    worker.query('sacct', '37835859')
    command, options = calls[0]
    assert '--clusters=create' in command and '--starttime=2026-10-06' in command
    assert options['timeout'] == 35


def test_query_timeout_preserved_without_retry(monkeypatch):
    calls = []
    def execute(command, **options):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, options['timeout'])
    monkeypatch.setattr(subprocess, 'run', execute)
    result = worker.query('sacct', '37835859')
    assert result['observation'] == 'timeout_status_unknown'
    assert len(calls) == 1


def test_refuses_local_or_login_execution(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='allocated CREATE'):
        worker.collect()
