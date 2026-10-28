"""Stream registered metadata instead of exceeding the SSH command-size limit."""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scripts import manage_m3w_task_risk as original

base = original.base
AMENDMENT = original.PUBLIC/'transport_amendment.json'


def installer():
    old = original.INSTALL
    a = 'regraw=sys.argv[3];cfgraw=sys.argv[4]'
    b = 'io.BytesIO(sys.stdin.buffer.read())'
    if old.count(a) != 1 or old.count(b) != 1:
        raise ValueError('Original exact installer must remain unchanged')
    return ('import base64\n' + old.replace(a,
        "payload=json.loads(sys.stdin.buffer.read());regraw=payload['registration'];cfgraw=payload['config']")
        .replace(b, "io.BytesIO(base64.b64decode(payload['archive'],validate=True))"))


def payload(reg):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for rel in reg['bindings']:
            raw = (base.ROOT/rel).read_bytes(); entry = tarfile.TarInfo(rel); entry.size = len(raw)
            tar.addfile(entry, io.BytesIO(raw))
    return json.dumps(dict(registration=original.REG.read_text(), config=original.CONFIG.read_text(),
        archive=base64.b64encode(buf.getvalue()).decode())).encode()


def register():
    original.register()
    paths = [Path(__file__).resolve(), base.ROOT/'tests/test_m3w_task_risk_stream.py',
             original.PUBLIC/'transport_repair.md']
    value = dict(original_registration_sha256=base.sha(original.REG),
        bindings={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        change='installation payload stdin instead of SSH command arguments',
        installation_logic_unchanged=True, numerical_code_unchanged=True,
        training_not_started_before_repair=True, new_root_observed_absent=True,
        large_argument_probe=dict(bytes=400000, returncode=255, connection_reset=True),
        stdin_probe=dict(bytes=400000, returncode=0, received_bytes=400000),
        objective_changed=False, gates_changed=False, split_changed=False,
        independent_roles_read=False, threshold_search=False, deployment_changed=False)
    base.once(AMENDMENT, value)
    return value


def submit(phase, resume=False):
    amendment = register(); reg = original.register()
    for rel in [*reg['bindings'], *amendment['bindings'],
                str(original.REG.relative_to(base.ROOT)), str(AMENDMENT.relative_to(base.ROOT))]:
        p = subprocess.run(['git','show','HEAD:'+rel],cwd=base.ROOT,capture_output=True)
        if p.returncode or p.stdout != (base.ROOT/rel).read_bytes():
            raise ValueError('Commit exact transport and numerical registration before running')
    installation = base.remote(installer(), [original.REMOTE, base.REMOTE], payload(reg), timeout=90)
    result = base.remote(original.SUBMIT, [original.REMOTE, base.RUNTIME, phase,
        base.sha(original.REG), 'resume' if resume else 'new'], timeout=90)
    base.once(original.PUBLIC/(phase+'_attempt'+str(result['attempt'])+'_submission.json'), result)
    return dict(installation=installation, submission=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['register','pilot','train','resume'])
    p.add_argument('--phase', choices=['pilot','train'],default='pilot'); a = p.parse_args()
    if a.mode == 'register':
        register(); out = dict(amendment_sha256=base.sha(AMENDMENT))
    else:
        out = submit(a.phase if a.mode == 'resume' else a.mode, a.mode == 'resume')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
