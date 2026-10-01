"""Read-only CREATE collection; never submits, retrains, or changes a decision."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import manage_m3w_selected_pool_create as manager
from scripts import run_m3w_selected_pool_accounting as run

CAP = 32 * 2**20


def verify_existing(path, text):
    if not path.exists():
        return False, False
    current = path.read_text()
    # Packet metadata is sorted JSON; dictionary key order is not a numerical difference.
    assert json.loads(current) == json.loads(text), 'Existing local values must match exactly'
    return True, current == text


def validate_bundle(bundle, manifest, registration_hash, job_id):
    complete = bundle['complete']
    assert bundle['registration_sha256'] == registration_hash
    assert bundle['manifest'] == manifest
    assert complete['job_id'] == str(job_id)
    assert complete['transfer_views'] == 216 and complete['exact_replay'] is True
    assert complete['parent_risk_checks'] == 2592
    assert complete['local_parity_groups'] == manifest['local_parity_groups']
    assert complete['new_parameter_updates'] == 0 and complete['policy_changed'] is False
    assert bundle['accounting']['returncode'] == 0
    lines = [s.split('|') for s in bundle['accounting']['stdout'].splitlines() if s.strip()]
    assert any(s[:3] == [str(job_id), 'COMPLETED', '0:0'] for s in lines)
    refs = complete['groups']
    assert len(refs) == len({r['group'] for r in refs}) == 216
    assert [r['group'] for r in refs] == [r['group'] for r in manifest['packets']]
    assert set(bundle['files']) == {r['group'] for r in refs}
    total = 0
    for ref in refs:
        name = ref['group']; assert re.fullmatch(r'[A-Za-z0-9_-]+', name)
        raw = bundle['files'][name].encode(); total += len(raw)
        assert len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256']
        doc = json.loads(raw)
        assert len(doc['rows']) == 3 and doc['parent_risk_checks'] == 12
        assert {r['mode'] for r in doc['rows']} == {'harm', 'reference', 'joint'}
        assert all(r['view'] == name and r['role'] == 'transfer' for r in doc['rows'])
    assert total <= CAP
    return total


def fetch_verified_bundle():
    run.registration()
    assert manager.registration() == json.loads((run.PUBLIC/'recovery_registration.json').read_text())
    manifest = json.loads((run.PUBLIC/'create_input_manifest.json').read_text())
    submission = json.loads((run.PUBLIC/'create_submission_receipt.json').read_text())
    job = submission['job_id']
    code = r'''
import base64,gzip,hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert root==pathlib.Path('/users/k24101830/m3w/european_selected_pool_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_selected_pool_v1'
receipt=json.loads((root/'submission_receipt.json').read_text());assert receipt['job_id']==p['job']
complete=json.loads((root/'complete.json').read_text())
assert complete['job_id']==p['job'] and complete['exact_replay']
files={};total=0
for ref in complete['groups']:
    name=ref['group'];assert pathlib.PurePath(name).name==name and '/' not in name
    raw=(root/'groups'/(name+'.json')).read_bytes();total+=len(raw);assert total<=p['cap']
    assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
    files[name]=raw.decode()
a=subprocess.run(['sacct','-j',p['job'],'-X','--noheader','--parsable2',
    '--format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList'],capture_output=True,text=True,timeout=30)
out=dict(complete=complete,files=files,manifest=json.loads((root/'input_manifest.json').read_text()),
    registration_sha256=hashlib.sha256((root/'recovery_registration.json').read_bytes()).hexdigest(),
    accounting=dict(returncode=a.returncode,stdout=a.stdout,stderr=a.stderr))
data=json.dumps(out).encode()
print(json.dumps({'payload':base64.b64encode(gzip.compress(data)).decode(),
    'expanded_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}))
'''
    payload = manager.remote(code, dict(root=manager.REMOTE, job=job, cap=CAP))
    assert payload['expanded_bytes'] <= 2 * CAP
    raw = gzip.decompress(base64.b64decode(payload['payload']))
    assert len(raw) == payload['expanded_bytes'] and hashlib.sha256(raw).hexdigest() == payload['sha256']
    bundle = json.loads(raw)
    total = validate_bundle(bundle, manifest, run.digest(run.PUBLIC/'recovery_registration.json'), job)
    return bundle, manifest, job, total


def main():
    cfg, _ = run.registration()
    bundle, manifest, job, total = fetch_verified_bundle()
    new_bytes = 0; existing = 0; byte_identical = 0
    for name, text in bundle['files'].items():
        path = run.PRIVATE/'transfer'/(name+'.json')
        present, same_bytes = verify_existing(path, text)
        if present:
            existing += 1; byte_identical += int(same_bytes)
        else:
            new_bytes += len(text.encode())
    assert existing >= manifest['local_parity_groups']
    # Preserve the original reserve, including the projected new lightweight outputs.
    assert shutil.disk_usage(run.PRIVATE).free >= cfg['disk_reserve_bytes'] + 32*2**20 + new_bytes
    refs = []
    for name, text in bundle['files'].items():
        path = run.PRIVATE/'transfer'/(name+'.json')
        if not path.exists(): run.immutable(path, json.loads(text))
        refs.append(run.base.artifact(path))
    run.immutable(run.PUBLIC/'transfer_manifest.json', dict(groups=refs, head_views=216, parent_risk_checks=2592))
    run.immutable(run.PUBLIC/'create_complete.json', bundle['complete'])
    run.immutable(run.PUBLIC/'transfer_replay.json', dict(all216_exact_replay=True,
        method='two allocated-node complete passes plus exact parsed-value local parity', job_id=job))
    run.immutable(run.PUBLIC/'transfer_runtime.json', dict(seconds=bundle['complete']['seconds'],
        timing_scope='CREATE_compute_and_replay_combined', result_source='fresh_run', job_id=job))
    receipt = dict(job_id=job, accounting=bundle['accounting'], output_bytes=total,
        local_parity_groups=manifest['local_parity_groups'], existing_groups_exact_values_verified=existing,
        existing_groups_also_byte_identical=byte_identical,
        total_groups=216, source_groups_already_verified=72, parent_risk_checks=2592,
        recovery_registration_sha256=run.digest(run.PUBLIC/'recovery_registration.json'),
        collector_sha256=run.digest(__file__), raw_arrays_collected=False, parameter_updates=0)
    run.immutable(run.PUBLIC/'create_collection_receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
