import base64
import io
import json
from types import SimpleNamespace

import pytest

from scripts import repair_m3w_temporal_create_root as repair


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'storage'/'european_temporal_auxiliary_v1'; root.mkdir(parents=True)
    alias = tmp_path/'users'/'european_temporal_auxiliary_v1'
    alias.parent.mkdir(); alias.symlink_to(root, target_is_directory=True)
    (root/'.owner.json').write_text('{"experiment":"european_temporal_auxiliary_v1"}')
    (root/'pilot_submission.json').write_text('{"job_id":"37790290"}')
    files = {}
    for rel in ('code/'+repair.SCRIPT, 'pilot_input_manifest.json', 'train_input_manifest.json'):
        before = ('old '+rel).encode(); after = ('new '+rel).encode()
        p = root/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(before)
        files[rel] = dict(before=base64.b64encode(before).decode(), after=base64.b64encode(after).decode())
    payload = dict(job_id='37790290', registration_sha256='a'*64, files=files)
    state = dict(status='PENDING', reason='(Priority)', commands=[])
    def command(argv, **_kwargs):
        state['commands'].append(argv)
        if argv[0] == 'squeue':
            result = state['status']+('|' + state['reason'] if argv[-1] == '%T|%R' else '')+'\n'
        elif argv == ['scontrol', 'hold', '37790290']:
            state['reason'] = '(JobHeldUser)'; result = ''
        elif argv == ['scontrol', 'release', '37790290']:
            state['reason'] = '(Priority)'; result = ''
        else: pytest.fail('Unexpected scheduler action: '+repr(argv))
        return SimpleNamespace(returncode=0, stdout=result, stderr='')
    monkeypatch.setattr(repair.manager.subprocess, 'run', command)
    def execute(program, value):
        monkeypatch.setattr('sys.argv', ['', str(alias)])
        monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(value)))
        exec(program.replace('/users/k24101830/m3w/european_temporal_auxiliary_v1', str(alias)), {})
    return root, payload, state, execute


def test_pending_job_repair_archives_bytes_then_explicit_release(tmp_path, monkeypatch, capsys):
    root, payload, state, execute = fixture(tmp_path, monkeypatch)
    execute(repair.REMOTE, payload); receipt = json.loads(capsys.readouterr().out)
    assert state['reason'] == '(JobHeldUser)' and receipt['job_id_preserved']
    for rel,pair in payload['files'].items():
        assert (root/rel).read_bytes() == base64.b64decode(pair['after'])
        assert (root/'execution_v2_before_ceph_fix'/rel).read_bytes() == base64.b64decode(pair['before'])
    before = list(state['commands'])
    with pytest.raises(SystemExit) as exc: execute(repair.REMOTE, payload)
    assert exc.value.code == 0 and state['commands'] == before
    assert json.loads(capsys.readouterr().out) == receipt
    execute(repair.RELEASE, receipt)
    result = json.loads(capsys.readouterr().out)
    assert result['no_new_submission'] and state['reason'] == '(Priority)'


@pytest.mark.parametrize('failure', ['running', 'changed_source', 'checkpoint', 'other_job', 'foreign_path'])
def test_bad_preconditions_never_hold_or_modify(tmp_path, monkeypatch, failure):
    root, payload, state, execute = fixture(tmp_path, monkeypatch)
    if failure == 'running': state['status'] = 'RUNNING'
    elif failure == 'changed_source': (root/'code'/repair.SCRIPT).write_bytes(b'foreign modification')
    elif failure == 'checkpoint':
        p=root/'data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1/heads/checkpoint.pt.gz'
        p.parent.mkdir(parents=True); p.write_bytes(b'existing work')
    elif failure == 'other_job': payload['job_id'] = '1'
    elif failure == 'foreign_path': payload['files']['../outside'] = payload['files'].pop('train_input_manifest.json')
    with pytest.raises(AssertionError): execute(repair.REMOTE, payload)
    assert not any(c[0] == 'scontrol' for c in state['commands'])
    assert not (root/'execution_v2_before_ceph_fix').exists()


def test_changed_weights_cannot_be_released(tmp_path, monkeypatch, capsys):
    root,payload,state,execute=fixture(tmp_path,monkeypatch)
    execute(repair.REMOTE,payload);receipt=json.loads(capsys.readouterr().out)
    (root/'code'/repair.SCRIPT).write_bytes(b'changed after repair')
    with pytest.raises(AssertionError):execute(repair.RELEASE,receipt)
    assert state['reason']=='(JobHeldUser)'
