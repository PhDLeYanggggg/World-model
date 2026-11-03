"""Hash-verified small checkpoint backup in the existing owned M3W CREATE root."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import socket
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'european_feature_duration_probe_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
HANDOFF = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'


def require_completed(public):
    summary, verification = public/'summary.json', public/'verification.json'
    if not summary.is_file() or not verification.is_file():
        raise ValueError('Complete18-model fitting replay required before completed backup')
    value, replay = json.loads(summary.read_text()), json.loads(verification.read_text())
    if (value.get('models') != 18 or not value.get('all18_old_budget_prefixes_exact')
            or not replay.get('all18_fitting_decompositions_exact')
            or replay.get('summary_sha256') != hashlib.sha256(summary.read_bytes()).hexdigest()):
        raise ValueError('Completed backup cannot promote partial or stale results')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed', action='store_true')
    args = parser.parse_args()
    receipt_path = PRIVATE/('create_completed_backup_receipt.json' if args.completed else 'create_backup_receipt.json')
    if receipt_path.exists():
        raise ValueError('Backup receipt exists; do not duplicate work')
    if args.completed:
        require_completed(PUBLIC)
    files = {str(p.relative_to(PRIVATE)):p for p in sorted((PRIVATE/'fits').glob('*.pt.gz'))}
    files.update({'first_full_refit.pt.gz':PRIVATE/'first_full_refit.pt.gz',
                  'train.log':PRIVATE/'train.log', 'registration.json':PUBLIC/'registration.json',
                  'protocol.md':PUBLIC/'protocol.md'})
    expected = 18 if args.completed else 7
    if len([n for n in files if n.startswith('fits/')]) != expected:
        raise ValueError('Keep the exact interrupted or complete checkpoint roster')
    if args.completed:
        files['resume_after_storage_recovery.log'] = PRIVATE/'resume_after_storage_recovery.log'
        for name in ('summary.json','complete.json','verification.json'):
            files[name] = PUBLIC/name
    raw = {name:path.read_bytes() for name,path in files.items()}
    manifest = {name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in raw.items()}
    if sum(v['bytes'] for v in manifest.values()) > 32*1024*1024:
        raise ValueError('This backup is limited to32MiB; no data caches or raw inputs')
    definition = dict(project='M3W',experiment=NAME,
        scope='complete_fitting_checkpoint_backup_not_remote_training' if args.completed else 'interrupted_checkpoint_backup_not_remote_training',
        files=manifest)
    backup_name = 'completed_backup_v1' if args.completed else 'recovery_backup_v1'
    blob = io.BytesIO()
    with zipfile.ZipFile(blob,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,value in raw.items():
            archive.writestr(name,value)
        archive.writestr('backup_manifest.json',json.dumps(definition,sort_keys=True))
    home = json.loads((HANDOFF/'compute_handoff_20260927.json').read_text())['paths']['user_home']
    ssh = json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    if not all(v in ssh for v in ('BatchMode=yes','IdentitiesOnly=yes','StrictHostKeyChecking=yes')):
        raise ValueError('Strict supplied SSH options required')
    address = '193.61.202.211'
    if address not in {v[4][0] for v in socket.getaddrinfo('hpc.create.kcl.ac.uk',22)}:
        raise ValueError('Official DNS endpoint must match')
    ssh = ssh[:1]+['-o','HostName='+address,'-o','HostKeyAlias=hpc.create.kcl.ac.uk']+ssh[1:]
    code = r'''
import hashlib,io,json,os,pathlib,sys,zipfile
home=pathlib.Path(sys.argv[1]);root=home/'m3w';name=sys.argv[2];backup_name=sys.argv[3]
assert name=='european_feature_duration_probe_v1'
assert backup_name in ('recovery_backup_v1','completed_backup_v1')
assert json.loads((root/'.m3w_owner.json').read_text())['project']=='M3W'
payload=sys.stdin.buffer.read(32*1024*1024+100000)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
 manifest=json.loads(z.read('backup_manifest.json'))
 assert manifest['project']=='M3W' and manifest['experiment']==name
 assert set(z.namelist())==set(manifest['files'])|{'backup_manifest.json'}
 assert len(set(z.namelist()))==len(z.namelist())
 required=sum(v['bytes'] for v in manifest['files'].values())
 assert int(os.getxattr(home,'ceph.quota.max_bytes'))-int(os.getxattr(home,'ceph.dir.rbytes'))>required+512*1024*1024
 destination=root/name/backup_name
 if destination.exists():
  assert json.loads((destination/'backup_manifest.json').read_text())==manifest
 else:
  assert not (root/name).is_symlink()
  destination.mkdir(parents=True)
 for relative,entry in manifest['files'].items():
  path=pathlib.PurePosixPath(relative)
  assert not path.is_absolute() and '..' not in path.parts
  target=destination/path
  raw=z.read(relative)
  assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert target.is_file() and not target.is_symlink() and target.read_bytes()==raw
  else:
   with target.open('xb') as stream:stream.write(raw)
 manifest_path=destination/'backup_manifest.json'
 if not manifest_path.exists():
  with manifest_path.open('x') as stream:json.dump(manifest,stream,sort_keys=True)
 for relative,entry in manifest['files'].items():
  assert hashlib.sha256((destination/relative).read_bytes()).hexdigest()==entry['sha256']
 print(json.dumps(dict(destination=str(destination),verified_files=len(manifest['files']),bytes=required,
  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
  all_files_read_back_exact=True,remote_training=False,jobs_submitted=0,simulation_touched=False)))
'''
    process = subprocess.run(ssh+[shlex.join(['python3','-c',code,home,NAME,backup_name])],
        input=blob.getvalue(),capture_output=True,timeout=60)
    if process.returncode:
        raise RuntimeError('Checkpoint backup failed; inspect SSH output without retrying a partial write: '+process.stderr.decode(errors='replace')[-1500:])
    result = json.loads(process.stdout)
    receipt = dict(observed_utc=datetime.now(timezone.utc).isoformat(),local_manifest=definition,remote=result,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),local_files_deleted=False)
    with receipt_path.open('x') as stream:
        json.dump(receipt,stream,indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='destination'}))


if __name__ == '__main__':
    main()
