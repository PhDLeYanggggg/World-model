"""Isolated alternate-node verification while the original recovery stays queued."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from scripts import manage_m3w_easy_harm_deviance as manager
from scripts.diagnose_m3w_easy_harm_node_portability import NAME

ROOT, HOME = manager.base.ROOT, manager.PUBLIC
REG = HOME/(NAME+'_registration.json')
SCRIPT = 'scripts/diagnose_m3w_easy_harm_node_portability.py'


def register():
    amendment = json.loads((HOME/'control_execution_amendment_v2.json').read_text())
    identities = sorted(k for k, v in amendment['references'].items() if not v['preserve_v1'])
    if len(identities) != 17:
        raise ValueError('Exactly17 frozen TRAIN references required')
    files = [ROOT/SCRIPT, ROOT/'scripts/diagnose_m3w_easy_harm_control.py']
    value = dict(training_registration_sha256=manager.base.sha(manager.REG),
        amendment_sha256=manager.base.sha(HOME/'control_execution_amendment_v2.json'),
        node='erc-hpc-comp188', identities=identities,
        code_bindings={str(p.relative_to(ROOT)):manager.base.sha(p) for p in files},
        manager_sha256=manager.base.sha(__file__),
        protocol_sha256=manager.base.sha(HOME/NAME/'protocol.md'),
        test_sha256=manager.base.sha(ROOT/'tests/test_m3w_easy_harm_node_portability.py'),
        references_changed=False, new_reference_exception=False, floating_tolerances_relaxed=False,
        verification_updates=34000, new_scientific_fits=0, existing_recovery_cancelled=False,
        validation_scored=False, independent_roles_read=False)
    manager.base.once(REG, value)
    return value


SUBMIT = r'''
import hashlib,io,json,pathlib,subprocess,sys,tarfile
root=pathlib.Path(sys.argv[1]);raw=sys.argv[2];reg=json.loads(raw);name='node_portability_v1'
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==reg['training_registration_sha256']
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_easy_harm_recovery_v2 import verify
from scripts.train_m3w_easy_harm_deviance import once,sha,quota,PUBLIC,PRIVATE
args,amendment,_=verify(root);assert sha(root/'control_execution_amendment_v2.json')==reg['amendment_sha256']
assert not (root/(name+'_intent.json')).exists(),'Inspect intent; no duplicate submission'
assert not (root/PUBLIC/'training_freeze.json').exists(),'Recovery already complete; no diagnostic needed'
resources=quota(args[0],args[1],args[2])
assert resources['cap_bytes']-resources['combined_checkpoint_bytes']>34*2**20,'Retain existing checkpoint cap'
q=subprocess.run(['squeue','-j','37835856','-h','-o','%T|%j'],capture_output=True,text=True,timeout=45)
assert q.returncode==0 and q.stdout.strip()=='PENDING|m3w_easy_recovery_v2_train'
seen=set()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
 for entry in tar.getmembers():
  assert entry.isfile() and entry.name in reg['code_bindings'] and entry.name not in seen
  data=tar.extractfile(entry).read();assert hashlib.sha256(data).hexdigest()==reg['code_bindings'][entry.name]
  path=root/'code'/entry.name;assert path.resolve().is_relative_to((root/'code').resolve())
  if path.exists():assert path.read_bytes()==data
  else:
   with path.open('xb') as f:f.write(data)
  seen.add(entry.name)
assert seen==set(reg['code_bindings'])
once(root/(name+'_registration.json'),reg)
runtime='/users/k24101830/m3w/easy_hurdle_runtime_v2/venv/bin/python'
lines=['#!/bin/bash -l','#SBATCH --job-name=m3w_easy_node_portability',
'#SBATCH --account=kcl','#SBATCH --partition=cpu','#SBATCH --qos=normal',
'#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --mem=8G','#SBATCH --time=01:00:00',
'#SBATCH --nodelist='+reg['node'],'#SBATCH --signal=B:TERM@120',
'#SBATCH --output='+str(root)+'/node-portability-%j.out',
'#SBATCH --error='+str(root)+'/node-portability-%j.err','set -euo pipefail',
'module load python/3.11.6-gcc-13.2.0','unset PYTHONPATH',
'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4',
'cd '+str(root/'code'),'exec '+runtime+' -m scripts.diagnose_m3w_easy_harm_node_portability --root '+str(root),'']
batch=root/(name+'.sbatch')
with batch.open('x') as f:f.write('\n'.join(lines))
once(root/(name+'_intent.json'),dict(registration_sha256=hashlib.sha256(raw.encode()).hexdigest(),resources=resources,
 existing_training_job_untouched='37835856',new_scientific_training=False))
try:
 p=subprocess.run(['sbatch','--hold','--parsable',str(batch)],capture_output=True,text=True,timeout=60)
 out=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 job=p.stdout.strip().split(';')[0]
 if p.returncode==0 and job.isdigit():out['job_id']=job
except subprocess.TimeoutExpired:out=dict(outcome='unknown_timeout_inspect_do_not_resubmit')
once(root/(name+'_submission.json'),out)
if 'job_id' in out:
 try:
  p=subprocess.run(['scontrol','release',out['job_id']],capture_output=True,text=True,timeout=60)
  out['release']=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except subprocess.TimeoutExpired:out['release']=dict(outcome='unknown_timeout_inspect')
 once(root/(name+'_release.json'),out['release'])
print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register','submit','status','collect'])
    args = parser.parse_args()
    if args.phase == 'register':
        row = register()
        print(json.dumps(dict(registered=True,node=row['node'],identities=len(row['identities']))))
        return
    if args.phase == 'submit':
        reg = register()
        files = [ROOT/p for p in reg['code_bindings']]
        checks = files+[REG,Path(__file__).resolve(),HOME/NAME/'protocol.md',ROOT/'tests/test_m3w_easy_harm_node_portability.py']
        for path in checks:
            if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))]) != path.read_bytes():
                raise ValueError('Commit bounded diagnostic before submission')
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w:gz') as tar:
            for path in files:
                raw = path.read_bytes(); item = tarfile.TarInfo(str(path.relative_to(ROOT))); item.size = len(raw)
                tar.addfile(item,io.BytesIO(raw))
        out = manager.base.remote(SUBMIT,[manager.REMOTE,REG.read_text()],buf.getvalue(),timeout=240)
        manager.base.once(HOME/(NAME+'_submission.json'),out)
    else:
        from scripts.manage_m3w_easy_harm_remaining_diagnostic import OBSERVE
        # Reuse the existing read-only collector with this isolated artifact prefix.
        code = OBSERVE.replace("name='control_diagnostic_v2'", "name='"+NAME+"'")
        code = code.replace("'remaining-diagnostic-'", "'node-portability-'")
        out = manager.base.remote(code,[manager.REMOTE],timeout=120)
        if args.phase == 'collect':
            assert out['complete'] and out['submission']['job_id']+'|COMPLETED|' in out['accounting']['stdout']
            assert '|0:0|' in out['accounting']['stdout']
            assert out['complete']['registration_sha256'] == manager.base.sha(REG)
            for row in out['rows']:
                manager.base.once(ROOT/row['path'],row['value'])
            manager.base.once(HOME/NAME/'complete.json',out['complete'])
            assert manager.base.sha(HOME/NAME/'complete.json') == out['complete_sha256']
            manager.base.once(HOME/(NAME+'_collection.json'),out)
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    main()
