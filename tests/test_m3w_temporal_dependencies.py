import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import repair_m3w_temporal_dependencies as deps
from scripts import retry_m3w_temporal_dependency_pilot as retry


def test_plan_only_adds_exact_missing_dependencies():
    before = dict(deps.CORE, joblib='1.6.0')
    assert deps.plan(before) == ['pandas==3.0.3', 'python-dateutil==2.9.0.post0', 'six==1.17.0']
    deps.validate(before, dict(before, **deps.REQUIRED))
    assert deps.plan(dict(before, **deps.REQUIRED)) == []


@pytest.mark.parametrize('change', ['core', 'existing_dependency', 'existing_other', 'extra', 'missing'])
def test_unregistered_environment_change_rejected(change):
    before = dict(deps.CORE, joblib='1.6.0'); after = dict(before, **deps.REQUIRED)
    if change == 'core':
        before['numpy'] = '0'; action = lambda: deps.plan(before)
    elif change == 'existing_dependency':
        before['pandas'] = '0'; action = lambda: deps.plan(before)
    else:
        if change == 'existing_other': after['joblib'] = '0'
        elif change == 'extra': after['extra'] = '1'
        else: del after['six']
        action = lambda: deps.validate(before, after)
    with pytest.raises(ValueError): action()


def test_install_and_import_guards_are_explicit():
    source = Path(deps.__file__).read_text()
    for item in ('SLURM_JOB_ID', 'LOCK_NB', '--no-deps', '--no-cache-dir', '--only-binary=:all:',
                 'https://pypi.org/simple', '--help', 'pip_check_passed'):
        assert item in source
    ast.parse(retry.REMOTE)


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'owned'; root.mkdir()
    logical = tmp_path/'logical'; logical.symlink_to(root, target_is_directory=True)
    (root/'.owner.json').write_text('{"experiment":"owned"}')
    (root/'pilot_submission.json').write_text('{"job_id":"37795593"}')
    (root/'pilot_submission_intent.json').write_text('{}')
    (root/'pilot-37795593.err').write_text("ModuleNotFoundError: No module named 'pandas'")
    (root/'pilot-37795593.out').write_bytes(b'')
    original='#!/bin/bash -l\n#SBATCH --time=02:00:00\n#SBATCH --mem=16G\n#SBATCH --cpus-per-task=4\nexec python -m scripts.train_m3w_temporal_auxiliary_portable pilot --root '+str(logical)+' --resume\n'
    (root/'pilot.sbatch').write_text(original)
    (root/'code/scripts').mkdir(parents=True)
    reg=dict(failed_job_id='37795593', bindings={retry.SCRIPT:hashlib.sha256(b'# bounded dependency code').hexdigest()})
    for phase in ('pilot','train'):
        p=root/(phase+'_input_manifest.json');p.write_text('{"code_bindings":{}}')
        reg[phase+'_manifest_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    payload=dict(registration=reg,registration_sha256='a'*64,dependency_script='# bounded dependency code')
    state=dict(accounting='FAILED|1:0',queue='',commands=[])
    def command(argv, **kwargs):
        state['commands'].append(argv)
        assert argv[0] in ('sacct','squeue','sbatch')
        stdout=state['accounting'] if argv[0]=='sacct' else state['queue'] if argv[0]=='squeue' else '12345\n'
        return SimpleNamespace(returncode=0,stdout=stdout,stderr='')
    monkeypatch.setattr(retry.subprocess,'run',command)
    monkeypatch.setattr('os.getxattr',lambda p,n:b'50000000000' if n=='ceph.quota.max_bytes' else b'0', raising=False)
    def execute():
        monkeypatch.setattr('sys.argv',['',str(logical)])
        monkeypatch.setattr('sys.stdin',io.StringIO(json.dumps(payload)))
        exec(retry.REMOTE.replace('/users/k24101830/m3w/european_temporal_auxiliary_v1',str(logical)),{})
    return root, state, execute, original


def test_exact_terminal_failure_submits_once_preserving_originals(tmp_path, monkeypatch, capsys):
    root,state,execute,original=fixture(tmp_path,monkeypatch)
    execute();out=json.loads(capsys.readouterr().out)
    assert out['submission']['job_id']=='12345' and not out['scientific_code_changed']
    assert (root/'execution_v5_dependency_failure/pilot.sbatch').read_text()==original
    assert (root/'pilot-37795593.err').exists()
    assert (root/'pilot.sbatch').read_text().endswith(original.split('\nexec ')[-1])
    with pytest.raises(AssertionError):execute()
    assert sum(c[0]=='sbatch' for c in state['commands'])==1


@pytest.mark.parametrize('failure',['running','other_error','active_job','checkpoint','heartbeat','full_train','stdout'])
def test_existing_work_or_wrong_failure_prevents_any_mutation(tmp_path, monkeypatch, failure):
    root,state,execute,_=fixture(tmp_path,monkeypatch)
    if failure=='running':state['accounting']='RUNNING|0:0'
    elif failure=='other_error':(root/'pilot-37795593.err').write_text('different failure')
    elif failure=='active_job':state['queue']='m3w_other_training'
    elif failure=='full_train':(root/'train_submission.json').write_text('{}')
    elif failure=='stdout':(root/'pilot-37795593.out').write_bytes(b'training')
    else:
        p=root/'data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1'/('checkpoint.pt.gz' if failure=='checkpoint' else 'heartbeat.json')
        p.parent.mkdir(parents=True);p.write_bytes(b'existing')
    with pytest.raises(AssertionError):execute()
    assert not any(c[0]=='sbatch' for c in state['commands'])
    assert (root/'pilot_submission.json').exists()
    assert not (root/'dependency_pilot_retry_intent.json').exists()
