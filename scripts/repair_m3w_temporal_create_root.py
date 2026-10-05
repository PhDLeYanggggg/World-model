"""Amend a verified pending M3W pilot; preserve its ID and all prior bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from scripts import manage_m3w_temporal_auxiliary_create as manager

SCRIPT = 'scripts/train_m3w_temporal_auxiliary_portable.py'
RECEIPT = manager.PUBLIC/'create_root_repair.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def request():
    manager.verify_registration()
    old_reg = json.loads((manager.PUBLIC/'create_port_registration_v2.json').read_text())
    new = (manager.ROOT/SCRIPT).read_bytes()
    old = new.replace(b"Path('/users/k24101830/m3w').resolve() and root.name", b"Path('/users/k24101830/m3w') and root.name")
    assert old != new and digest(old) == old_reg['bindings'][SCRIPT]
    files = {'code/'+SCRIPT: dict(before=base64.b64encode(old).decode(), after=base64.b64encode(new).decode())}
    for phase in ('pilot', 'train'):
        path = manager.PRIVATE/('create_'+phase+'_input_manifest.json')
        archived = manager.PRIVATE/('create_'+phase+'_input_manifest_v2.json')
        raw = (archived if archived.exists() else path).read_bytes()
        assert digest(raw) == json.loads((manager.PUBLIC/('create_'+phase+'_transfer.json')).read_text())['manifest_sha256']
        manifest = json.loads(raw)
        assert manifest['code_bindings'][SCRIPT] == digest(old)
        manifest['code_bindings'][SCRIPT] = digest(new)
        changed = (json.dumps(manifest, indent=2)+'\n').encode()
        files[phase+'_input_manifest.json'] = dict(before=base64.b64encode(raw).decode(), after=base64.b64encode(changed).decode())
    return dict(job_id='37790290', registration_sha256=manager.sha(manager.PORT_REGISTRATION), files=files)


REMOTE = r'''
import base64,hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
root=root.resolve();a=json.loads(sys.stdin.read());job=a['job_id'];assert job=='37790290'
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_temporal_auxiliary_v1'
assert json.loads((root/'pilot_submission.json').read_text())['job_id']==job
assert not (root/'train_submission.json').exists()
allowed={'code/scripts/train_m3w_temporal_auxiliary_portable.py','pilot_input_manifest.json','train_input_manifest.json'}
assert set(a['files'])==allowed
values={rel:{k:base64.b64decode(v,validate=True) for k,v in pair.items()} for rel,pair in a['files'].items()}
assert all(set(pair)=={'before','after'} and pair['before']!=pair['after'] for pair in values.values())
receipt=root/'execution_v3_repair.json';archive=root/'execution_v2_before_ceph_fix'
checks={rel:{k:hashlib.sha256(v).hexdigest() for k,v in pair.items()} for rel,pair in values.items()}
if receipt.exists():
 out=json.loads(receipt.read_text());assert out['file_hashes']==checks and out['job_id']==job and out['registration_sha256']==a['registration_sha256']
 for rel,pair in values.items():
  assert (root/rel).read_bytes()==pair['after'] and (archive/rel).read_bytes()==pair['before']
 print(json.dumps(out));sys.exit(0)
for rel,pair in values.items():
 p=(root/rel).resolve();assert p.is_relative_to(root) and p.read_bytes() in pair.values()
private=root/'data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1'
assert not list(private.rglob('*.pt.gz'))
public=root/'outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1'
assert not (public/'pilot.json').exists() and not (public/'training_freeze.json').exists()
q=subprocess.run(['squeue','-j',job,'-h','-o','%T'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='PENDING','Only a verified pending, unstarted pilot can be amended'
q=subprocess.run(['scontrol','hold',job],capture_output=True,text=True,timeout=20);assert q.returncode==0,q.stderr
q=subprocess.run(['squeue','-j',job,'-h','-o','%T|%R'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='PENDING|(JobHeldUser)'
intent=root/'execution_v3_repair_intent.json'
expected=dict(job_id=job,registration_sha256=a['registration_sha256'],file_hashes=checks)
if intent.exists():assert json.loads(intent.read_text())==expected
else:intent.write_text(json.dumps(expected,indent=2)+'\n')
for rel,pair in values.items():
 p=root/rel;old=archive/rel;old.parent.mkdir(parents=True,exist_ok=True)
 if old.exists():assert old.read_bytes()==pair['before']
 else:old.write_bytes(pair['before'])
 assert p.read_bytes() in pair.values()
 temp=p.with_name(p.name+'.v3-repair-tmp')
 with temp.open('wb') as f:f.write(pair['after']);f.flush();os.fsync(f.fileno())
 os.replace(temp,p);assert hashlib.sha256(p.read_bytes()).hexdigest()==checks[rel]['after']
out=dict(**expected,state='repaired_held_waiting_verified_release',job_id_preserved=True,
 scientific_training_code_unchanged=True,optimizer_updates_before_repair=0,packets_modified=False,
 prior_bytes_archived=True,validation_read=False,independent_roles_read=False)
receipt.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''


RELEASE = r'''
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);assert root==pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
root=root.resolve();a=json.loads(sys.stdin.read());r=json.loads((root/'execution_v3_repair.json').read_text());assert r==a
job=r['job_id'];assert job=='37790290'
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_temporal_auxiliary_v1'
assert json.loads((root/'pilot_submission.json').read_text())['job_id']==job
for rel,hashes in r['file_hashes'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==hashes['after']
q=subprocess.run(['squeue','-j',job,'-h','-o','%T|%R'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='PENDING|(JobHeldUser)','Inspect manually after any ambiguous release; do not duplicate'
q=subprocess.run(['scontrol','release',job],capture_output=True,text=True,timeout=20);assert q.returncode==0,q.stderr
out=dict(job_id=job,release_returncode=q.returncode,original_job_preserved=True,no_new_submission=True)
(root/'execution_v3_release.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''


def repair():
    a = request()
    result = manager.remote(REMOTE, [manager.REMOTE], json.dumps(a).encode(), timeout=100)
    assert result['job_id'] == a['job_id'] and result['registration_sha256'] == a['registration_sha256']
    for phase in ('pilot', 'train'):
        pair = a['files'][phase+'_input_manifest.json']
        before, after = (base64.b64decode(pair[k]) for k in ('before', 'after'))
        path = manager.PRIVATE/('create_'+phase+'_input_manifest.json')
        archived = manager.PRIVATE/('create_'+phase+'_input_manifest_v2.json')
        if archived.exists(): assert archived.read_bytes() == before
        else: archived.write_bytes(before)
        assert path.read_bytes() in (before, after)
        path.write_bytes(after)
    manager.once(RECEIPT, result)
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=['repair', 'release'])
    a = p.parse_args()
    if a.phase == 'repair': repair()
    else:
        manager.verify_registration()
        r = json.loads(RECEIPT.read_text())
        result = manager.remote(RELEASE, [manager.REMOTE], json.dumps(r).encode())
        manager.once(manager.PUBLIC/'create_root_repair_release.json', result)
        print(json.dumps(result))


if __name__ == '__main__': main()
