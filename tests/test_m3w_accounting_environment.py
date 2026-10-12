from types import SimpleNamespace
import subprocess

import pytest

from scripts import probe_m3w_create_accounting_environment as probe


@pytest.mark.parametrize('kind', ['sbatch', 'scancel', 'scontrol', 'shell'])
def test_no_mutations_or_broad_queries(kind):
    with pytest.raises(ValueError):
        probe.command(kind)


def test_fixed_jobs_and_only_cluster_flag_difference():
    explicit = probe.command('sacct')
    local = probe.command('sacct', False)
    assert local == [arg for arg in explicit if arg != '--clusters=create']
    assert explicit[explicit.index('-j') + 1] == ','.join(probe.JOBS)
    assert '--starttime=2026-10-06' in explicit


def test_queries_preserve_real_errors_without_retry(monkeypatch):
    calls = []
    def execute(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout='', stderr='database unavailable')
    monkeypatch.setattr(subprocess, 'run', execute)
    result = probe.query('sacct')
    assert result['returncode'] == 1 and result['stderr'] == 'database unavailable'
    assert len(calls) == 1 and calls[0][1]['timeout'] == 25


def test_timeout_does_not_infer_terminal_status(monkeypatch):
    def execute(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', execute)
    assert probe.query('sacct')['observation'] == 'query_unavailable_status_unknown'


def test_local_execution_rejected(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='allocated CREATE'):
        probe.collect('system')
