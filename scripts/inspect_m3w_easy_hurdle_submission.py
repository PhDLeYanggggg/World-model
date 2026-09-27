"""Read remote submission intent/receipt after an SSH failure, without submitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from scripts.export_m3w_easy_hurdle_create import remote
from scripts.prepare_m3w_create_runtime import PRIVATE, HANDOFF


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['pilot','train'],required=True);a=p.parse_args()
    ssh=json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    home=json.loads((PRIVATE/'remote_input_manifest.json').read_text())['remote_path']
    code=r'''
import json,pathlib,subprocess,sys
home=pathlib.Path(sys.argv[1]);phase=sys.argv[2]
assert phase in ('pilot','train') and json.loads((home/'.owner.json').read_text())['experiment']=='european_easy_hurdle_v1'
r={}
for k in ('intent','receipt'):
    p=home/('submit_'+phase+'_'+k+'.json');r[k]=json.loads(p.read_text()) if p.exists() else None
p=home/('run_'+phase+'.sh');r['script_exists']=p.exists()
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=20)
assert q.returncode==0
r['matching_jobs']=[s for s in q.stdout.splitlines() if 'm3w_easy_head_' in s]
print(json.dumps(r))
'''
    result=remote(ssh,code,[home,a.phase])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=PRIVATE/f'create_submit_safety_{a.phase}_{stamp}.json'
    with out.open('x') as f:
        f.write(json.dumps(dict(observed_utc=stamp,remote_modified=False,**result),indent=2)+'\n')
    print(json.dumps(dict(phase=a.phase,**result,observation_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))


if __name__=='__main__':
    main()
