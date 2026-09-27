"""Upload and submit one owned audit job after the full training receipt exists."""
import argparse
import hashlib
import json
import shlex
import subprocess
from scripts.prepare_m3w_create_runtime import ROOT, PRIVATE, HANDOFF


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',type=int,choices=range(1,4),default=1);a=p.parse_args()
    suffix='' if a.attempt==1 else '_attempt'+str(a.attempt)
    output=PRIVATE/('create_verification_submission'+suffix+'.json')
    assert not output.exists(),'Existing/uncertain local submission requires inspection'
    for n in range(1,a.attempt):
        prev=PRIVATE/('create_verification_submission'+('' if n==1 else '_attempt'+str(n))+'.json')
        assert json.loads(prev.read_text())['returncode'] in (255,None)
    home=json.loads((PRIVATE/'remote_input_manifest.json').read_text())['remote_path']
    source=(ROOT/'scripts/verify_m3w_easy_hurdle_portable_training.py').read_text()
    script=f'''#!/bin/bash -l
#SBATCH --job-name=m3w_easy_head_verify
#SBATCH --partition=cpu
#SBATCH --account=kcl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=verify-%j.out
#SBATCH --error=verify-%j.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
module load python/3.11.6-gcc-13.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset PYTHONPATH
../easy_hurdle_runtime_v2/venv/bin/python code/scripts/verify_m3w_easy_hurdle_portable_training.py --home "$PWD"
'''
    code=r'''
import hashlib,json,pathlib,subprocess,sys
p=json.loads(sys.stdin.read());home=pathlib.Path(p['home'])
assert json.loads((home/'.owner.json').read_text())['experiment']=='european_easy_hurdle_v1'
receipt=json.loads((home/'training_complete.json').read_text());assert receipt['groups']==108 and receipt['complete']
q=subprocess.run(['sacct','-j',receipt['environment']['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='COMPLETED|0:0'
assert not (home/'verification_submit_intent.json').exists() and not (home/'verification_submit_receipt.json').exists()
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and 'm3w_easy_head_' not in q.stdout
for relative,text in [('code/scripts/verify_m3w_easy_hurdle_portable_training.py',p['source']),('run_verify.sh',p['script'])]:
    path=home/relative
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
(home/'verification_submit_intent.json').write_text(json.dumps({'verifier_sha256':hashlib.sha256(p['source'].encode()).hexdigest()})+'\n')
r=subprocess.run(['sbatch','--parsable','run_verify.sh'],cwd=home,capture_output=True,text=True,timeout=30)
out=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
if r.returncode==0:out['job_id']=r.stdout.strip().split(';')[0];assert out['job_id'].isdigit()
(home/'verification_submit_receipt.json').write_text(json.dumps(out)+'\n');print(json.dumps(out))
'''
    ssh=json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    try:
        r=subprocess.run(ssh+[shlex.join(['/usr/bin/python3','-c',code])],input=json.dumps(dict(home=home,source=source,script=script)),capture_output=True,text=True,timeout=60)
        result=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except subprocess.TimeoutExpired:result=dict(returncode=None,observation_timeout=True)
    with output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(returncode=result['returncode'],receipt_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
