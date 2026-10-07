"""Register and recover the last17 paired fits without touching110 accepted fits."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scripts import manage_m3w_easy_harm_deviance as manager
from scripts import train_m3w_easy_harm_recovery_v2 as runner

HOME, ROOT = manager.PUBLIC, manager.base.ROOT
REG = HOME/runner.AMENDMENT
CODE = 'scripts/train_m3w_easy_harm_recovery_v2.py'


def register():
    old = json.loads((HOME/'control_execution_amendment.json').read_text())
    complete = json.loads((HOME/'control_diagnostic_v2/complete.json').read_text())
    observed = json.loads((HOME/'partial_training_observation_20261006T1256Z.json').read_text())
    manifest = json.loads((manager.base.PRIVATE/'create_train_input_manifest.json').read_text())
    if (complete['summary']['original_new_exact'] != 17 or complete['summary']['metadata_failures']
            or complete['summary']['original_historical_exact'] != 0 or len(observed['fits']) != 110):
        raise ValueError('Complete prior diagnosis and110 preserved receipts required')
    first = manifest['heads'][0]
    references = {manager.run.key(first,'quadratic'): dict(identity=first['identity'],preserve_v1=True,
        reference_checkpoint=old['reference_checkpoint'])}
    proof_paths = [HOME/'control_diagnostic_v1.json',HOME/'control_diagnostic_v2/complete.json',
                   HOME/'partial_training_observation_20261006T1256Z.json']
    for ref in complete['results']:
        path = ROOT/ref['path']
        if manager.base.sha(path) != ref['sha256']:
            raise ValueError('Changed diagnosis')
        row = json.loads(path.read_text())
        if not row['comparisons']['old_direct_vs_new_direct']['all_control_fields_exact']:
            raise ValueError('Exact original/new replay required')
        references[row['name']] = dict(identity=row['identity'],preserve_v1=False,
            reference_checkpoint=row['checkpoints']['old_direct'],adopt_checkpoint=row['checkpoints']['new_direct'])
        proof_paths.append(path)
    expected = {manager.run.key(r,'quadratic') for r in manifest['heads'][0::4]}
    if set(references) != expected or len(references) != 18:
        raise ValueError('Do not expand the18 known execution exceptions')
    files = [ROOT/CODE,Path(__file__).resolve(),ROOT/'tests/test_m3w_easy_harm_recovery_v2.py',HOME/'control_recovery_v2.md']
    reg = dict(training_registration_sha256=manager.base.sha(manager.REG),
        previous_amendment_sha256=manager.base.sha(HOME/'control_execution_amendment.json'),
        references=references,proofs=[dict(path=str(p.relative_to(ROOT)),sha256=manager.base.sha(p)) for p in proof_paths],
        preserved_fit_refs=[{k:r[k] for k in ('path','sha256')} for r in observed['fits']],
        preserved_shards=observed['shards'],execution_bindings={CODE:manager.base.sha(ROOT/CODE)},
        bindings={str(p.relative_to(ROOT)):manager.base.sha(p) for p in files},
        new_compute_node='erc-hpc-comp186', resubmitted_shards=[0],
        historical_controls_exact=54, original_implementation_replay_controls_exact=18,
        adopted_diagnostic_quadratic_fits=17,new_candidate_fits=17,steps_per_fit=2000,
        scientific_training_unchanged=True, historical_bitwise_reproduction_complete=False,
        floating_tolerances_relaxed=False,validation_scored=False,independent_roles_read=False,
        all144_required=True,stage5c_executed=False,smc_enabled=False)
    manager.base.once(REG,reg)
    return reg


SUBMIT = r'''
import hashlib,io,json,os,pathlib,shutil,subprocess,sys,tarfile
root=pathlib.Path(sys.argv[1]);rawreg=sys.argv[2];reg=json.loads(rawreg)
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==reg['training_registration_sha256']
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_easy_harm_deviance import verify,quota,sha,once,PUBLIC
r,source,cfg,_,_=verify(root);resources=quota(r,source,cfg)
assert not (root/'recovery_v2_intent.json').exists(),'Inspect existing intent; never duplicate'
assert not (root/PUBLIC/'training_freeze.json').exists()
assert sha(root/'control_execution_amendment.json')==reg['previous_amendment_sha256']
for ref in reg['proofs']+reg['preserved_fit_refs']:
 p=(root/ref['path']).resolve();assert p.is_relative_to((root/PUBLIC).resolve()) and sha(p)==ref['sha256']
for ref in reg['preserved_shards'].values():assert sha(root/ref['path'])==ref['sha256']
for ref in reg['references'].values():
 for name in ('reference_checkpoint','adopt_checkpoint'):
  if name in ref:
   cp=ref[name];p=(root/cp['path']).resolve();assert p.is_relative_to((root/'data/stage_cvpr2027_experiments'/root.name).resolve())
   assert sha(p)==cp['sha256'] and p.stat().st_size==cp['bytes']
for job,state in [('37815216_0','FAILED|1:0'),('37817976','COMPLETED|0:0'),('37814169_1','COMPLETED|0:0'),('37815216_2','COMPLETED|0:0'),('37815216_3','COMPLETED|0:0')]:
 p=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
 assert p.returncode==0 and p.stdout.strip()==state
p=subprocess.run(['squeue','-j','37815222','-h','-o','%T|%j'],capture_output=True,text=True,timeout=20)
assert p.returncode==0 and p.stdout.strip()=='PENDING|m3w_easy_verified_join'
seen={}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for entry in tar.getmembers():
  assert entry.isfile() and entry.name in reg['execution_bindings'] and entry.name not in seen
  raw=tar.extractfile(entry).read();assert hashlib.sha256(raw).hexdigest()==reg['execution_bindings'][entry.name]
  path=root/'code'/entry.name;assert path.resolve().is_relative_to((root/'code').resolve())
  if path.exists():assert path.read_bytes()==raw
  else:
   with path.open('xb') as f:f.write(raw)
  seen[entry.name]=hashlib.sha256(raw).hexdigest()
assert seen==reg['execution_bindings']
once(root/'control_execution_amendment_v2.json',reg)
from scripts.train_m3w_easy_harm_recovery_v2 import verify as verify_recovery
verify_recovery(root)
archive=root/'execution_archive_v2';archive.mkdir(exist_ok=False)
files=['train_submission.json','join_submission.json','verified_resume_receipt.json',
       'verified_train.sbatch','verified_join.sbatch','verified-train-37815216_0.out','verified-train-37815216_0.err']
old={}
for name in files:
 p=root/name;shutil.copy2(p,archive/name);assert sha(p)==sha(archive/name);old[name]=sha(p)
once(root/'recovery_v2_intent.json',dict(amendment_sha256=hashlib.sha256(rawreg.encode()).hexdigest(),resources=resources,archived=old))
runtime='/users/k24101830/m3w/easy_hurdle_runtime_v2/venv/bin/python'
def dispatch(kind,extra):
 join=kind=='join';cpu=1 if join else 4
 lines=['#!/bin/bash -l','#SBATCH --job-name=m3w_easy_recovery_v2_'+kind,'#SBATCH --account=kcl','#SBATCH --partition=cpu','#SBATCH --qos=normal',
 '#SBATCH --ntasks=1','#SBATCH --cpus-per-task='+str(cpu),'#SBATCH --mem='+('2G' if join else '16G'),
 '#SBATCH --time='+('00:20:00' if join else '12:00:00'),
 '#SBATCH --output='+str(root)+'/recovery-v2-'+kind+'-%A_%a.out','#SBATCH --error='+str(root)+'/recovery-v2-'+kind+'-%A_%a.err','#SBATCH --signal=B:TERM@120']
 if not join:lines.append('#SBATCH --nodelist='+reg['new_compute_node'])
 lines+=['set -euo pipefail','module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
 'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS='+str(cpu)+' OPENBLAS_NUM_THREADS='+str(cpu)+' MKL_NUM_THREADS='+str(cpu)+' NUMEXPR_NUM_THREADS='+str(cpu),
 'cd '+str(root/'code'),'exec '+runtime+' -m scripts.train_m3w_easy_harm_recovery_v2 '+kind+' --root '+str(root),'']
 path=root/('recovery_v2_'+kind+'.sbatch')
 with path.open('x') as f:f.write('\n'.join(lines))
 try:
  p=subprocess.run(['sbatch','--parsable','--hold',*extra,str(path)],capture_output=True,text=True,timeout=30)
  row=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,registration_sha256=reg['training_registration_sha256'],execution_amendment_sha256=hashlib.sha256(rawreg.encode()).hexdigest())
  job=p.stdout.strip().split(';')[0]
  if p.returncode==0 and job.isdigit():row['job_id']=job
 except subprocess.TimeoutExpired:row=dict(outcome='unknown_timeout_inspect_do_not_resubmit')
 once(root/('recovery_v2_'+kind+'_submission.json'),row);return row
out={'train':dispatch('train',['--array=0'])}
if 'job_id' in out['train']:out['join']=dispatch('join',['--dependency=afterok:'+out['train']['job_id']])
if 'job_id' in out.get('join',{}):
 for kind in ('train','join'):
  path=root/(kind+'_submission.json');temp=path.with_suffix('.recovery_v2')
  with temp.open('x') as f:json.dump(out[kind],f,indent=2);f.write('\n')
  os.replace(temp,path)
 p=subprocess.run(['squeue','-j','37815222','-h','-o','%T|%j'],capture_output=True,text=True,timeout=20)
 assert p.returncode==0 and p.stdout.strip()=='PENDING|m3w_easy_verified_join'
 p=subprocess.run(['scancel','37815222'],capture_output=True,text=True,timeout=20);assert p.returncode==0
 out['obsolete_pending_join_retired']='37815222';out['release']=[]
 for kind in ('train','join'):
  p=subprocess.run(['scontrol','release',out[kind]['job_id']],capture_output=True,text=True,timeout=20)
  out['release'].append(dict(kind=kind,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
once(root/'recovery_v2_receipt.json',out)
print(json.dumps(out))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','submit']); a=p.parse_args()
    reg=register()
    if a.phase=='register':
        print(json.dumps(dict(references=len(reg['references']),preserved=110,new_training_fits=17,sha256=manager.base.sha(REG))))
        return
    for rel in [*reg['bindings'],str(REG.relative_to(ROOT))]:
        if subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)!=(ROOT/rel).read_bytes():
            raise ValueError('Commit explicit execution amendment before submitting')
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        raw=(ROOT/CODE).read_bytes();entry=tarfile.TarInfo(CODE);entry.size=len(raw);tar.addfile(entry,io.BytesIO(raw))
    result=manager.base.remote(SUBMIT,[manager.REMOTE,REG.read_text()],buf.getvalue(),timeout=180)
    manager.base.once(HOME/'recovery_v2_submission.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
