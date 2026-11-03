"""Read-only recovery snapshot: own checkpoints, disk and approved CREATE root."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_feature_duration_probe_v1'
HANDOFF = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'


def main():
    now = datetime.now(timezone.utc)
    destination = PRIVATE/('recovery_observation_'+now.strftime('%Y%m%dT%H%M%SZ')+'.json')
    if destination.exists():
        raise ValueError('Do not overwrite observations')
    handoff = HANDOFF/'compute_handoff_20260927.json'
    if hashlib.sha256(handoff.read_bytes()).hexdigest() != 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9':
        raise ValueError('Approved owned-path handoff changed')
    home = json.loads(handoff.read_text())['paths']['user_home']
    ssh = json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    if not all(v in ssh for v in ('BatchMode=yes','IdentitiesOnly=yes','StrictHostKeyChecking=yes')):
        raise ValueError('Retain the authorized strict SSH options')
    address = '193.61.202.211'
    if address not in {v[4][0] for v in socket.getaddrinfo('hpc.create.kcl.ac.uk',22)}:
        raise ValueError('Use only the currently resolved official endpoint')
    ssh = ssh[:1]+['-o','HostName='+address,'-o','HostKeyAlias=hpc.create.kcl.ac.uk']+ssh[1:]
    code = r'''
import datetime,hashlib,json,os,pathlib,subprocess,sys
home=pathlib.Path(sys.argv[1]); root=home/'m3w'
out=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands={},metadata=[])
for name,argv in [('hostname',['hostname']),('queue',['squeue','--clusters=create','--me','--noheader','--format=%i|%j|%P|%T|%R']),('cpu_partition',['scontrol','--clusters=create','show','partition','cpu','--oneliner']),('filesystem',['df','-Pk',str(home)])]:
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=10)
  out['commands'][name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except subprocess.TimeoutExpired:
  out['commands'][name]=dict(returncode=None,observation_timeout=True)
out['quota']={}
for key in ('ceph.quota.max_bytes','ceph.dir.rbytes'):
 try:out['quota'][key]=os.getxattr(home,key).decode()
 except OSError as e:out['quota'][key]=dict(unavailable=True,errno=e.errno)
owner=root/'.m3w_owner.json'
out['owned_root_verified']=owner.is_file() and json.loads(owner.read_text()).get('project')=='M3W'
if out['owned_root_verified']:
 names=('training_freeze.json','cache_freeze.json','complete.json','runtime_receipt.json','receipt.json')
 out['directories']=[]
 for folder in sorted(root.iterdir()):
  if not folder.is_dir() or folder.is_symlink() or not folder.resolve().is_relative_to(root.resolve()):continue
  out['directories'].append(folder.name)
  for name in names:
   path=folder/name
   if path.is_file() and not path.is_symlink() and path.stat().st_size<2*1024*1024:
    raw=path.read_bytes()
    try:value=json.loads(raw)
    except (ValueError,UnicodeError):continue
    if isinstance(value,dict):out['metadata'].append(dict(directory=folder.name,file=name,sha256=hashlib.sha256(raw).hexdigest(),keys=sorted(value)[:30]))
out.update(remote_modified=False,jobs_submitted=0,scientific_compute=False,inputs_or_labels_read=False)
print(json.dumps(out))
'''
    try:
        process = subprocess.run(ssh+[shlex.join(['python3','-c',code,home])],capture_output=True,text=True,timeout=60)
        response = dict(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)
        if process.returncode == 0:
            response['decoded'] = json.loads(process.stdout)
    except subprocess.TimeoutExpired:
        response = dict(returncode=None,observation_timeout=True)
    disk = shutil.disk_usage(ROOT)
    value = dict(observed_utc=now.isoformat(),local_disk_free_bytes=disk.free,
        private_probe_bytes=sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()),
        completed_fit_receipts=len(list((ROOT/'outputs/publication_readiness_2026_09/european_feature_duration_probe_v1/fits').glob('*.json'))),
        training_log_sha256=hashlib.sha256((PRIVATE/'train.log').read_bytes()).hexdigest(),
        remote=response,remote_modified=False,jobs_submitted=0)
    with destination.open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')
    decoded = response.get('decoded',{})
    print(json.dumps(dict(receipt=str(destination.relative_to(ROOT)),local_disk_free_bytes=disk.free,
        completed_fit_receipts=value['completed_fit_receipts'],remote_returncode=response['returncode'],
        owned_root_verified=decoded.get('owned_root_verified'),metadata_receipts=len(decoded.get('metadata',[])),
        command_returncodes={k:v['returncode'] for k,v in decoded.get('commands',{}).items()},
        remote_modified=False,jobs_submitted=0)))


if __name__ == '__main__':
    main()
