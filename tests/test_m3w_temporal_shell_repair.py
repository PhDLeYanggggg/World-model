import hashlib
import inspect
import io
import json
from types import SimpleNamespace

import pytest

from scripts import manage_m3w_temporal_auxiliary_create as manager
from scripts import repair_m3w_temporal_create_shell as repair


def test_generated_submission_initializes_login_environment():
    source = inspect.getsource(manager.submit)
    assert "['#!/bin/bash -l'," in source
    assert "'module load python/3.11.6-gcc-13.2.0'" in source
    assert '--root {root} --resume' in source


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'storage'; root.mkdir()
    alias = tmp_path/'users'; alias.symlink_to(root, target_is_directory=True)
    (root/'.owner.json').write_text('{"experiment":"european_temporal_auxiliary_v1"}')
    manifests = {}
    code = root/'code'/'runner.py'; code.parent.mkdir(); code.write_bytes(b'unchanged code')
    digest = hashlib.sha256(code.read_bytes()).hexdigest()
    for phase in ('pilot', 'train'):
        p = root/(phase+'_input_manifest.json')
        p.write_text(json.dumps(dict(code_bindings={'runner.py': digest})))
        manifests[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'pilot_submission.json').write_text('{"job_id":"37790290"}')
    (root/'pilot_submission_intent.json').write_text(json.dumps(dict(manifest_sha256=manifests['pilot_input_manifest.json'])))
    (root/'pilot.sbatch').write_text('#!/bin/bash\nmodule load python/3.11.6-gcc-13.2.0\npython run --resume\n')
    (root/'pilot-37790290.err').write_text('slurm_script: line 14: module: command not found\n')
    (root/'pilot-37790290.out').write_bytes(b'')
    payload = dict(failed_job_id='37790290', registration_sha256='a'*64,
                   pilot_manifest_sha256=manifests['pilot_input_manifest.json'], input_manifest_hashes=manifests)
    state = dict(accounting='FAILED|127:0', queue='', commands=[])

    def command(argv, **_kwargs):
        state['commands'].append(argv)
        assert argv[0] in ('sacct', 'squeue')
        return SimpleNamespace(returncode=0, stdout=state['accounting'] if argv[0]=='sacct' else state['queue'], stderr='')

    monkeypatch.setattr(manager.subprocess, 'run', command)

    def execute():
        monkeypatch.setattr('sys.argv', ['', str(alias)])
        monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(payload)))
        exec(repair.REMOTE.replace('/users/k24101830/m3w/european_temporal_auxiliary_v1', str(alias)), {})

    return root, payload, state, execute


def test_exact_failure_archived_without_submitting_or_changing_inputs(tmp_path, monkeypatch, capsys):
    root, payload, state, execute = fixture(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    execute(); result = json.loads(capsys.readouterr().out)
    assert not result['new_job_submitted'] and result['optimizer_updates_before_retry'] == 0
    for name in ('pilot_submission.json', 'pilot_submission_intent.json', 'pilot.sbatch'):
        assert not (root/name).exists()
        assert (root/'execution_v3_shell_failure'/name).read_bytes() == before[name]
    for name in ('pilot_input_manifest.json', 'train_input_manifest.json', 'pilot-37790290.err'):
        assert (root/name).read_bytes() == before[name]
    commands = list(state['commands'])
    with pytest.raises(SystemExit) as exc: execute()
    assert exc.value.code == 0 and commands == state['commands']
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize('failure', ['running', 'other_error', 'active_job', 'stdout', 'checkpoint', 'heartbeat',
                                     'full_submission', 'changed_code', 'manifest', 'other_id'])
def test_wrong_failure_or_existing_work_stops_before_mutation(tmp_path, monkeypatch, failure):
    root, payload, state, execute = fixture(tmp_path, monkeypatch)
    if failure == 'running': state['accounting'] = 'RUNNING|0:0'
    elif failure == 'other_error': (root/'pilot-37790290.err').write_text('unrelated error')
    elif failure == 'active_job': state['queue'] = 'm3w_temporal_pilot\n'
    elif failure == 'stdout': (root/'pilot-37790290.out').write_bytes(b'training')
    elif failure in ('checkpoint', 'heartbeat'):
        p = root/'data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1'/('checkpoint.pt.gz' if failure=='checkpoint' else 'heartbeat.json')
        p.parent.mkdir(parents=True); p.write_bytes(b'existing work')
    elif failure == 'full_submission': (root/'train_submission.json').write_text('{}')
    elif failure == 'changed_code': (root/'code/runner.py').write_bytes(b'changed')
    elif failure == 'manifest': payload['input_manifest_hashes']['train_input_manifest.json'] = 'b'*64
    elif failure == 'other_id': payload['failed_job_id'] = '1'
    with pytest.raises(AssertionError): execute()
    assert (root/'pilot_submission.json').exists() and not (root/'execution_v3_shell_failure').exists()
