"""Verify a completed remote receipt and checkpoint bytes; publish no row data."""
import argparse
import hashlib
import json
from scripts import run_m3w_easy_hurdle as run
from scripts.export_m3w_easy_hurdle_create import remote
from scripts.prepare_m3w_create_runtime import HANDOFF


def validate(result, manifest, phase):
    expected=json.dumps({k:v for k,v in manifest.items() if k!='remote_path'},indent=2)+'\n'
    r=result['receipt']
    assert r['manifest_sha256']==hashlib.sha256(expected.encode()).hexdigest()
    assert not r['held_outcomes_used'] and not r['independent_roles_read']
    assert result['scheduler_state']=='COMPLETED|0:0'
    expected_groups=1 if phase=='pilot' else 108
    assert r['groups']==expected_groups and len(r['artifacts'])==expected_groups*2
    assert r['pilot']==(phase=='pilot') and r['complete']==(phase=='train')
    expected_keys={(g['group'],arm) for g in manifest['groups'][:expected_groups] for arm in run.api.ARMS}
    assert {(a['group'],a['arm']) for a in r['artifacts']}==expected_keys
    assert result['verified_checkpoint_files']==len(r['artifacts'])
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['pilot','train'],required=True);a=p.parse_args()
    manifest=json.loads((run.PRIVATE/'remote_input_manifest.json').read_text())
    ssh=json.loads((HANDOFF/'observations.json').read_text())['ssh_arguments']
    code=r'''
import hashlib,json,pathlib,subprocess,sys
home=pathlib.Path(sys.argv[1]);phase=sys.argv[2];assert phase in ('pilot','train')
assert json.loads((home/'.owner.json').read_text())['experiment']=='european_easy_hurdle_v1'
p=home/('pilot.json' if phase=='pilot' else 'training_complete.json');r=json.loads(p.read_text())
job=r['environment']['job_id'];assert job.isdigit()
q=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and q.stdout.strip()=='COMPLETED|0:0'
assert hashlib.sha256((home/'input_manifest.json').read_bytes()).hexdigest()==r['manifest_sha256']
size=0
for ref in r['artifacts']:
    rel=pathlib.PurePosixPath(ref['path']);assert not rel.is_absolute() and '..' not in rel.parts
    assert rel==pathlib.PurePosixPath('heads')/ref['group']/ref['arm']/'checkpoint.pt.gz'
    pth=home/ref['path'];assert hashlib.sha256(pth.read_bytes()).hexdigest()==ref['sha256'];size+=pth.stat().st_size
print(json.dumps({'receipt':r,'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'scheduler_state':q.stdout.strip(),'verified_checkpoint_files':len(r['artifacts']),'checkpoint_bytes':size}))
'''
    result=remote(ssh,code,[manifest['remote_path'],a.phase]);r=validate(result,manifest,a.phase)
    out=dict(result_source='fresh_run_training_with_fresh_remote_hash_verification',**result,
        registration=run.base.artifact(run.PUBLIC/'registration.json'),
        updates_per_head=100 if a.phase=='pilot' else 2000,
        total_model_updates=200 if a.phase=='pilot' else 432000,
        held_development_readout='not_run',independent_confirmation='not_run',deployment_changed=False,
        stage5c_executed=False,smc_enabled=False)
    name='create_pilot_result.json' if a.phase=='pilot' else 'create_training_freeze.json'
    run.base.immutable_json(run.PUBLIC/name,out)
    print(json.dumps(dict(groups=r['groups'],heads=len(r['artifacts']),fit_seconds=r['seconds'],
        checkpoint_bytes=result['checkpoint_bytes'],public_receipt=name)))


if __name__=='__main__':
    main()
