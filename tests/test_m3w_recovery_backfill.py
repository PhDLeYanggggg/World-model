import ast
import subprocess
from types import SimpleNamespace

from scripts.allow_m3w_recovery_backfill import REMOTE


def update_function(monkeypatch, implementation):
    monkeypatch.setattr(subprocess, 'run', implementation)
    tree = ast.parse(REMOTE)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'update')
    scope = {'subprocess': subprocess}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<registered update>', 'exec'), scope)
    return scope['update']


def test_timeout_is_unknown_not_success_or_retry(monkeypatch):
    calls = []

    def timeout(args, **kwargs):
        calls.append((args, kwargs))
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])

    result = update_function(monkeypatch, timeout)()
    assert result['returncode'] is None
    assert result['outcome'] == 'unknown_inspect_do_not_retry'
    assert len(calls) == 1
    assert calls[0][0] == ['scontrol', 'update', 'JobId=37835856', 'TimeMin=01:00:00']
    assert calls[0][1]['timeout'] == 60


def test_scheduler_rejection_is_preserved(monkeypatch):
    result = update_function(monkeypatch, lambda *a, **k: SimpleNamespace(
        returncode=1, stdout='', stderr='Access/permission denied'))()
    assert result['returncode'] == 1
    assert result['stderr'] == 'Access/permission denied'


def test_intent_precedes_update_and_no_resubmission():
    assert REMOTE.index('once(intent,') < REMOTE.index('result=update()')
    assert "assert not intent.exists()" in REMOTE
    assert "'sbatch'" not in REMOTE and "'scancel'" not in REMOTE
    assert 'scientific_contract_changed=False' in REMOTE
