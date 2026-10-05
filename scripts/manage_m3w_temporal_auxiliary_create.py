"""Isolated CREATE transport/submission for the already registered TRAIN experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'european_temporal_auxiliary_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
REMOTE = '/users/k24101830/m3w/'+NAME
RUNTIME = '/users/k24101830/m3w/easy_hurdle_runtime_v2'
HANDOFF = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923/observations.json'
INPUT_CAP = 4*2**30
PORT_REGISTRATION = PUBLIC/'create_port_registration_v3.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Immutable receipt mismatch: '+str(path)
    else:
        with path.open('x') as f: f.write(json.dumps(value, indent=2)+'\n')


def ssh_args():
    args = json.loads(HANDOFF.read_text())['ssh_arguments']
    assert 'StrictHostKeyChecking=yes' in args and 'BatchMode=yes' in args
    return args


def remote(code, args=(), payload=b'', timeout=80):
    # Submission is never automatically retried after an ambiguous outcome.
    p = subprocess.run(ssh_args()+[shlex.join(['/usr/bin/python3', '-c', code, *args])],
                       input=payload, capture_output=True, timeout=timeout)
    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-3000:])
    return json.loads(p.stdout)


def record_observation(value, prefix):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = PRIVATE/(prefix+'_'+stamp+'.json')
    once(path, dict(observed_utc=stamp, **value))
    return str(path.relative_to(ROOT))


def inspect():
    code = r'''
import json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);rt=pathlib.Path(sys.argv[2]);home=root.parent.parent
assert json.loads((root.parent/'.m3w_owner.json').read_text())['project']=='M3W'
r=json.loads((rt/'runtime_receipt.json').read_text())
assert json.loads((rt/'.m3w_runtime_owner.json').read_text())['project']=='M3W'
assert r['checkpoint_resume_exact'] and not r['research_training']
assert r['torch'].split('+')[0]=='2.12.0' and r['numpy']=='2.4.6'
a=subprocess.run(['sacct','-j',r['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0','Runtime accounting must verify'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%P|%T'],capture_output=True,text=True,timeout=20);assert q.returncode==0
limit=int(os.getxattr(home,'ceph.quota.max_bytes'));used=int(os.getxattr(home,'ceph.dir.rbytes'))
print(json.dumps(dict(runtime_receipt=r,runtime_accounting=a.stdout.strip(),quota_limit=limit,quota_used=used,
 quota_free=limit-used,m3w_queue=[s for s in q.stdout.splitlines() if '|m3w_' in s],
 other_jobs_untouched=True,experiment_exists=root.exists(),remote_modified=False)))
'''
    out = remote(code, [REMOTE, RUNTIME])
    ref = record_observation(out, 'create_preflight')
    print(json.dumps(dict(receipt=ref, **out)), flush=True)
    return out


def register():
    from scripts.export_m3w_easy_hurdle_create import closure
    paths = closure(ROOT, ['scripts.train_m3w_temporal_auxiliary_portable'])
    paths += [Path(__file__).resolve(), ROOT/'tests/test_m3w_temporal_auxiliary_portable.py',
              PUBLIC/'create_port_protocol.md', PUBLIC/'create_port_filename_fix.md',
              PUBLIC/'create_port_root_fix.md',
              ROOT/'scripts/repair_m3w_temporal_create_root.py',
              ROOT/'tests/test_m3w_temporal_root_repair.py']
    reg = dict(experiment=NAME, execution_revision=3,
        previous_port_registration_sha256=sha(PUBLIC/'create_port_registration_v2.json'),
        original_registration_sha256=sha(PUBLIC/'registration.json'),
        original_config_sha256=sha(CONFIG), bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        input_cap_bytes=INPUT_CAP, checkpoint_cap_bytes=268435456, disk_reserve_bytes=10737418240,
        runtime=RUNTIME, execution_change_only=True, optimizer_unchanged=True,
        split_unchanged=True, validation_selection_unchanged=True, independent_roles_read=False)
    once(PORT_REGISTRATION, reg)
    return reg


def verify_registration():
    reg = json.loads(PORT_REGISTRATION.read_text())
    for rel, digest in reg['bindings'].items(): assert sha(ROOT/rel) == digest
    assert sha(CONFIG) == reg['original_config_sha256']
    assert sha(PUBLIC/'registration.json') == reg['original_registration_sha256']
    for rel in [*reg['bindings'], str(PORT_REGISTRATION.relative_to(ROOT))]:
        q = subprocess.run(['git', 'show', 'HEAD:'+rel], cwd=ROOT, capture_output=True)
        assert q.returncode == 0 and q.stdout == (ROOT/rel).read_bytes(), 'Commit registration before transport'
    return reg


def setup(reg):
    bundle = io.BytesIO()
    bindings = {k:v for k,v in reg['bindings'].items() if k.endswith('.py') and not k.startswith('tests/')
                and k != 'scripts/manage_m3w_temporal_auxiliary_create.py'}
    with tarfile.open(fileobj=bundle, mode='w:gz') as tar:
        for rel in bindings:
            raw = (ROOT/rel).read_bytes(); entry = tarfile.TarInfo(rel); entry.size = len(raw)
            tar.addfile(entry, io.BytesIO(raw))
    code = r'''
import hashlib,io,json,os,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]);assert root.parent==pathlib.Path('/users/k24101830/m3w')
assert json.loads((root.parent/'.m3w_owner.json').read_text())['project']=='M3W'
owner=json.loads(sys.argv[2]);root.mkdir(exist_ok=True);p=root/'.owner.json'
if p.exists():assert json.loads(p.read_text())==owner
else:p.write_text(json.dumps(owner)+'\n')
checks={}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for m in tar.getmembers():
  rel=pathlib.PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
  raw=tar.extractfile(m).read();p=root/'code'/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw)
  checks[m.name]=hashlib.sha256(raw).hexdigest()
print(json.dumps(checks))
'''
    owner = dict(experiment=NAME, registration_sha256=reg['original_registration_sha256'],
                 port_registration_sha256=sha(PUBLIC/'create_port_registration.json'))
    assert remote(code, [REMOTE, json.dumps(owner)], bundle.getvalue()) == bindings
    return bindings


RECEIVER = r'''
import hashlib,json,os,pathlib,re,sys
root=pathlib.Path(sys.argv[1]);reserve=int(sys.argv[2]);cap=int(sys.argv[3]);home=root.parent.parent
assert root.parent==pathlib.Path('/users/k24101830/m3w') and root.name=='european_temporal_auxiliary_v1'
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
stream=sys.stdin.buffer;print(json.dumps({'ready':True}),flush=True)
while True:
 line=stream.readline(4097)
 if not line:break
 assert len(line)<=4096 and line.endswith(b'\n');header=json.loads(line);name=header['group'];size=header['bytes']
 assert re.fullmatch(r'[A-Za-z0-9_-]+',name) and 0<size<=512*2**20
 chunks=[];left=size
 while left:
  chunk=stream.read(min(left,1024*1024));assert chunk;chunks.append(chunk);left-=len(chunk)
 content=b''.join(chunks);digest=hashlib.sha256(content).hexdigest();assert digest==header['sha256']
 p=root/'inputs'/(name+'.npz');p.parent.mkdir(exist_ok=True)
 if p.exists():assert p.read_bytes()==content
 else:
  free=int(os.getxattr(home,'ceph.quota.max_bytes'))-int(os.getxattr(home,'ceph.dir.rbytes'))
  assert free>=reserve+size+268435456+2097152,'Personal quota reserve violated'
  assert sum(f.stat().st_size for f in p.parent.glob('*.npz'))+size<=cap,'Registered input cap exceeded'
  tmp=p.with_suffix('.tmp')
  with tmp.open('wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
 print(json.dumps(dict(group=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())),flush=True)
'''


def export(phase):
    reg = verify_registration(); preflight = inspect(); bindings = setup(reg)
    from scripts import run_m3w_temporal_auxiliary as run
    from scripts.train_m3w_temporal_auxiliary_portable import packet, unpack
    from scripts.m3w_packet_stream import PacketStream
    cfg = json.loads(CONFIG.read_text()); run.api.torch.set_num_threads(4); run.api.torch.set_num_interop_threads(1)
    # The inherited loader has exposed source assets; only its original TRAIN
    # slices leave this process. No validation rows or new independent roles.
    print(json.dumps(dict(state='loading_verified_TRAIN_sources', phase=phase)), flush=True)
    _, _, data, jobs, oid, _, _, _ = run.parent.inner.old.load()
    heads = []; sent = {}; stream = PacketStream(ssh_args(), REMOTE)
    stream.command = ssh_args()+[shlex.join(['/usr/bin/python3', '-c', RECEIVER, REMOTE,
                                           str(cfg['disk_reserve_bytes']), str(INPUT_CAP)])]
    with stream:
        for identity, args in run.inputs(data, jobs, oid):
            group = identity['group']
            content = packet(args); digest = hashlib.sha256(content).hexdigest()
            run.api.core.exact(args, unpack(content))
            if group in sent:
                assert sent[group]['sha256'] == digest, 'Seeds must share identical TRAIN arrays'
                ref = sent[group]
            else:
                ref = stream.send(group, content); sent[group] = ref
                once(PRIVATE/'create_packets'/(group+'.json'), ref)
                print(json.dumps(dict(state='TRAIN_packet_verified', group=group, bytes=len(content),
                                      unique_packets=len(sent))), flush=True)
            heads.append(dict(**ref, identity=identity))
            if phase == 'pilot': break
    assert len(heads) == (1 if phase == 'pilot' else cfg['source_heads'])
    manifest = dict(heads=heads, code_bindings=bindings, config_sha256=sha(CONFIG),
        registration_sha256=reg['original_registration_sha256'], input_role='source_TRAIN_only',
        validation_rows_transferred=False, independent_roles_read=False,
        runtime_receipt=preflight['runtime_receipt'])
    code = r'''
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);a=json.loads(sys.stdin.read())
for name,raw in a.items():
 assert name in ('config.json','pilot_input_manifest.json','train_input_manifest.json');p=root/name
 if p.exists():assert p.read_text()==raw
 else:p.write_text(raw)
print(json.dumps({'complete':True}))
'''
    raw = json.dumps(manifest, indent=2)+'\n'
    remote(code, [REMOTE], json.dumps({'config.json': CONFIG.read_text(), phase+'_input_manifest.json':raw}).encode())
    once(PRIVATE/('create_'+phase+'_input_manifest.json'), manifest)
    once(PUBLIC/('create_'+phase+'_transfer.json'), dict(source='fresh_run_TRAIN_only_lossless_transport',
        unique_packets=len(sent), source_heads=len(heads), bytes=sum(r['bytes'] for r in sent.values()),
        manifest_sha256=hashlib.sha256(raw.encode()).hexdigest(), all_packets_hash_verified=True,
        optimizer_updates=0, validation_rows_transferred=False, independent_roles_read=False))


def submit(phase):
    verify_registration(); inspect()
    manifest = PRIVATE/('create_'+phase+'_input_manifest.json')
    digest = sha(manifest)
    code = r'''
import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];expected=sys.argv[3];rt=sys.argv[4]
assert root.parent==pathlib.Path('/users/k24101830/m3w') and root.name=='european_temporal_auxiliary_v1'
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
assert hashlib.sha256((root/(phase+'_input_manifest.json')).read_bytes()).hexdigest()==expected
intent=root/(phase+'_submission_intent.json');receipt=root/(phase+'_submission.json')
assert not intent.exists() and not receipt.exists(),'Existing submission intent: inspect, never duplicate'
q=subprocess.run(['squeue','--me','-h','-o','%j'],capture_output=True,text=True,timeout=20);assert q.returncode==0
assert not any(s.startswith('m3w_temporal_') for s in q.stdout.splitlines())
if phase=='train':
 pilot=json.loads((root/'pilot_submission.json').read_text());job=pilot['job_id']
 ac=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
 assert ac.returncode==0 and ac.stdout.strip()=='COMPLETED|0:0'
 out=json.loads((root/'outputs/publication_readiness_2026_09'/root.name/'pilot.json').read_text())
 assert out['exact_interrupted_resume'] and out['matched_draws'] and out['local_time_feasible']
 assert out['peak_RSS_bytes']<14*2**30
home=root.parent.parent;free=int(os.getxattr(home,'ceph.quota.max_bytes'))-int(os.getxattr(home,'ceph.dir.rbytes'))
assert free>=10737418240+268435456+2097152
wall='02:00:00' if phase=='pilot' else '12:00:00'
script='\n'.join(['#!/bin/bash',f'#SBATCH --job-name=m3w_temporal_{phase}', '#SBATCH --account=kcl',
 '#SBATCH --partition=cpu','#SBATCH --qos=normal','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4',
 '#SBATCH --mem=16G',f'#SBATCH --time={wall}',f'#SBATCH --output={root}/{phase}-%j.out',
 f'#SBATCH --error={root}/{phase}-%j.err','#SBATCH --signal=B:TERM@120','set -euo pipefail',
 'module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
 'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4',
 f'cd {root}/code',f'exec {rt}/venv/bin/python -m scripts.train_m3w_temporal_auxiliary_portable {phase} --root {root} --resume',''])
path=root/(phase+'.sbatch');path.write_text(script)
with intent.open('x') as f:json.dump(dict(phase=phase,manifest_sha256=expected,quota_free=free),f)
try:
 p=subprocess.run(['sbatch','--parsable',str(path)],capture_output=True,text=True,timeout=30)
 out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 job=p.stdout.strip().split(';')[0]
 if p.returncode==0 and job.isdigit():out['job_id']=job
except subprocess.TimeoutExpired:out=dict(outcome='unknown_timeout_do_not_resubmit')
receipt.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''
    out = remote(code, [REMOTE, phase, digest, RUNTIME])
    once(PRIVATE/('create_'+phase+'_submission.json'), out)
    print(json.dumps(out), flush=True)


def status():
    code = r'''
import json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);out={}
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_temporal_auxiliary_v1'
for phase in ('pilot','train'):
 p=root/(phase+'_submission.json')
 if not p.exists():continue
 r=json.loads(p.read_text());job=r.get('job_id');row={'submission':r}
 if job:
  for name,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%M|%R']),('accounting',['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,Elapsed,ExitCode,MaxRSS'])]:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);row[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
  for suffix in ('out','err'):
   p=root/(phase+'-'+job+'.'+suffix);row[suffix+'_tail']=p.read_text()[-6000:] if p.exists() else None
 out[phase]=row
private=root/'data/stage_cvpr2027_experiments'/root.name;public=root/'outputs/publication_readiness_2026_09'/root.name
for name,p in [('heartbeat',private/'heartbeat.json'),('pilot_result',public/'pilot.json'),('training_freeze',public/'training_freeze.json')]:
 out[name]=json.loads(p.read_text()) if p.exists() else None
out['checkpoint_count']=len(list(private.rglob('*.pt.gz')))
print(json.dumps(out))
'''
    out = remote(code, [REMOTE]); receipt = record_observation(out, 'create_status')
    print(json.dumps(dict(receipt=receipt, **out)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['inspect', 'register', 'export-pilot', 'export-train', 'submit-pilot', 'submit-train', 'status'])
    a = p.parse_args()
    if a.phase == 'inspect': inspect()
    elif a.phase == 'register': register(); print('CREATE-only execution amendment registered; no optimizer updates')
    elif a.phase.startswith('export-'): export(a.phase.split('-')[1])
    elif a.phase.startswith('submit-'): submit(a.phase.split('-')[1])
    else: status()


if __name__ == '__main__':
    main()
