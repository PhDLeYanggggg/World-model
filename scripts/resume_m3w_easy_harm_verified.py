"""Preserve completed work and resume only the three unfinished owned shards."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scripts import manage_m3w_easy_harm_deviance as manager

HOME = manager.PUBLIC
REG = HOME/'control_execution_amendment.json'
CODE = 'scripts/train_m3w_easy_harm_verified.py'


def register():
    diag = json.loads((HOME/'control_diagnostic_v1.json').read_text())
    observed = json.loads((HOME/'partial_training_observation_20261006T102747Z.json').read_text())
    manifest = json.loads((manager.base.PRIVATE/'create_train_input_manifest.json').read_text())
    assert diag['job_id'] == '37814380'
    for name in ('old_direct_vs_new_direct','new_direct_vs_failed_new',
                 'old_from_new_pilot_vs_failed_new','old_direct_vs_old_from_new_pilot'):
        assert diag['comparisons'][name]['all_control_fields_exact']
    assert not diag['comparisons']['old_direct_vs_historical_old']['all_control_fields_exact']
    paths = [CODE,'scripts/resume_m3w_easy_harm_verified.py',
             'tests/test_m3w_easy_harm_verified.py',str((HOME/'control_replay_amendment.md').relative_to(manager.base.ROOT))]
    row = dict(training_registration_sha256=manager.base.sha(manager.REG),
        diagnostic_sha256=manager.base.sha(HOME/'control_diagnostic_v1.json'),
        exception_identity=manifest['heads'][0]['identity'],
        reference_checkpoint=diag['checkpoints']['old_direct'],
        execution_bindings={CODE:manager.base.sha(manager.base.ROOT/CODE)},
        bindings={p:manager.base.sha(manager.base.ROOT/p) for p in paths},
        preserved_array_job_id='37814169', preserved_shard=1,
        preserved_fit_refs=[r['receipt'] for r in observed['partial_fits']],
        # Obtain this exact byte hash from the already verified remote receipt.
        preserved_shard_sha256=json.loads((HOME/'resume_inventory.json').read_text())['preserved_shard_sha256'],
        resubmitted_shards=[0,2,3], new_compute_node='erc-hpc-comp186',
        scientific_training_unchanged=True, historical_reference_requirement_amended=True,
        floating_tolerances_relaxed=False, alternative_is_exact_original_trainer_replay=True,
        validation_scored=False, independent_roles_read=False,
        historical_failure_remains_reported=True, all144_required=True,
        main_fits=144,updates_per_fit=2000,stage5c_executed=False,smc_enabled=False)
    manager.base.once(REG,row)
    return row


INVENTORY = r'''
import hashlib,json,pathlib,subprocess
r=pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((r/'.owner.json').read_text())['experiment']==r.name
home=r/'outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1'
out={'files':{},'jobs':{}}
for name in ('train_submission.json','join_submission.json','train_submission_intent.json','join_submission_intent.json','train_release.json','train.sbatch','join.sbatch'):
 p=r/name;out['files'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
out['preserved_shard_sha256']=hashlib.sha256((home/'shards/1.json').read_bytes()).hexdigest()
for job in ('37814169','37814170','37814380'):
 q=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=JobID,State,ExitCode'],capture_output=True,text=True,timeout=20);assert q.returncode==0;out['jobs'][job]=q.stdout
print(json.dumps(out))
'''


SUBMIT = r'''
import hashlib,io,json,os,pathlib,shutil,subprocess,sys,tarfile
root=pathlib.Path(sys.argv[1]);rawreg=sys.argv[2];reg=json.loads(rawreg);inventory=json.loads(sys.argv[3])
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==reg['training_registration_sha256']
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_easy_harm_deviance import verify,quota,sha,once,PUBLIC
r,source,cfg,_,_=verify(root);resources=quota(r,source,cfg)
assert not (root/'verified_resume_intent.json').exists(),'Existing intent: inspect, never duplicate'
assert sha(root/PUBLIC/'control_diagnostic_v1.json')==reg['diagnostic_sha256']
assert sha(root/PUBLIC/'shards/1.json')==reg['preserved_shard_sha256']
assert sha(root/reg['reference_checkpoint']['path'])==reg['reference_checkpoint']['sha256']
for name,ref in inventory['files'].items():assert sha(root/name)==ref['sha256']
for job,expected in [('37814169_0','FAILED|1:0'),('37814169_1','COMPLETED|0:0'),('37814380','COMPLETED|0:0')]:
 p=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20);assert p.returncode==0 and p.stdout.strip()==expected
for job in ('37814169_2','37814169_3','37814170'):
 p=subprocess.run(['squeue','-j',job,'-h','-o','%T|%j'],capture_output=True,text=True,timeout=20);assert p.returncode==0 and p.stdout.startswith('PENDING|m3w_easy_deviance_')
seen={}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for member in tar.getmembers():
  assert member.isfile() and member.name in reg['execution_bindings'] and member.name not in seen
  raw=tar.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==reg['execution_bindings'][member.name]
  path=root/'code'/member.name;assert path.resolve().is_relative_to((root/'code').resolve())
  if path.exists():assert path.read_bytes()==raw
  else:
   with path.open('xb') as f:f.write(raw)
  seen[member.name]=hashlib.sha256(raw).hexdigest()
assert seen==reg['execution_bindings']
once(root/'control_execution_amendment.json',reg)
archive=root/'execution_archive_v1';archive.mkdir(exist_ok=False)
for name in inventory['files']:
 shutil.copy2(root/name,archive/name);assert sha(archive/name)==inventory['files'][name]['sha256']
for name in ('train-37814169_0.out','train-37814169_0.err'):
 shutil.copy2(root/name,archive/name)
once(root/'verified_resume_intent.json',dict(registration_sha256=hashlib.sha256(rawreg.encode()).hexdigest(),resources=resources,previous_files=inventory['files']))
runtime='/users/k24101830/m3w/easy_hurdle_runtime_v2/venv/bin/python'
def dispatch(kind,extra):
 join=kind=='join';cpu=1 if join else 4;wall='00:20:00' if join else '12:00:00';mem='2G' if join else '16G'
 command=runtime+' -m scripts.train_m3w_easy_harm_verified '+kind+' --root '+str(root)
 if not join:command+=' --shard "$SLURM_ARRAY_TASK_ID"'
 lines=['#!/bin/bash -l','#SBATCH --job-name=m3w_easy_verified_'+kind,'#SBATCH --account=kcl','#SBATCH --partition=cpu','#SBATCH --qos=normal','#SBATCH --ntasks=1',
 '#SBATCH --cpus-per-task='+str(cpu),'#SBATCH --mem='+mem,'#SBATCH --time='+wall,
 '#SBATCH --output='+str(root)+'/verified-'+kind+'-%A_%a.out','#SBATCH --error='+str(root)+'/verified-'+kind+'-%A_%a.err','#SBATCH --signal=B:TERM@120']
 if not join:lines.append('#SBATCH --nodelist='+reg['new_compute_node'])
 lines+=['set -euo pipefail','module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
 'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS='+str(cpu)+' OPENBLAS_NUM_THREADS='+str(cpu)+' MKL_NUM_THREADS='+str(cpu)+' NUMEXPR_NUM_THREADS='+str(cpu),
 'cd '+str(root/'code'),'exec '+command,'']
 path=root/('verified_'+kind+'.sbatch')
 with path.open('x') as f:f.write('\n'.join(lines))
 p=subprocess.run(['sbatch','--parsable','--hold',*extra,str(path)],capture_output=True,text=True,timeout=30)
 row=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,registration_sha256=reg['training_registration_sha256'],execution_amendment_sha256=hashlib.sha256(rawreg.encode()).hexdigest())
 job=p.stdout.strip().split(';')[0]
 if p.returncode==0 and job.isdigit():row['job_id']=job
 once(root/('verified_'+kind+'_submission.json'),row);return row
out={'train':dispatch('train',['--array=0,2,3%3'])}
if 'job_id' in out['train']:
 out['join']=dispatch('join',['--dependency=afterok:'+out['train']['job_id']])
if 'job_id' in out.get('join',{}):
 for kind in ('train','join'):
  path=root/(kind+'_submission.json');tmp=path.with_suffix('.replacement')
  with tmp.open('x') as f:json.dump(out[kind],f,indent=2);f.write('\n')
  os.replace(tmp,path)
 out['old_pending_cancelled']=[]
 for job in ('37814169_2','37814169_3','37814170'):
  p=subprocess.run(['scancel',job],capture_output=True,text=True,timeout=20);assert p.returncode==0;out['old_pending_cancelled'].append(job)
 out['release']=[]
 for kind in ('train','join'):
  p=subprocess.run(['scontrol','release',out[kind]['job_id']],capture_output=True,text=True,timeout=20);out['release'].append({'kind':kind,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
once(root/'verified_resume_receipt.json',out)
print(json.dumps(out))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['inventory','register','submit']); args = p.parse_args()
    if args.mode == 'inventory':
        value = manager.base.remote(INVENTORY,[],timeout=90)
        manager.base.once(HOME/'resume_inventory.json',value)
    elif args.mode == 'register': value = register()
    else:
        reg = register()
        for rel in [*reg['bindings'],str(REG.relative_to(manager.base.ROOT))]:
            saved = subprocess.check_output(['git','show','HEAD:'+rel],cwd=manager.base.ROOT)
            assert saved == (manager.base.ROOT/rel).read_bytes(),'Commit amendment before execution'
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload,mode='w:gz') as tar:
            raw = (manager.base.ROOT/CODE).read_bytes();item=tarfile.TarInfo(CODE);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
        value = manager.base.remote(SUBMIT,[manager.REMOTE,REG.read_text(),(HOME/'resume_inventory.json').read_text()],payload.getvalue(),timeout=90)
        manager.base.once(HOME/'verified_resume_submission.json',value)
    print(json.dumps(value,indent=2))


if __name__ == '__main__':
    main()
