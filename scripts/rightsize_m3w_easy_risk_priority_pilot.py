"""One-time, pending-only pilot walltime correction; no resubmission or fit change."""
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_risk_priority import PUBLIC, PRIVATE, REMOTE, registration, digest, immutable
from scripts.manage_m3w_easy_gradient_diagnostic import call_remote


def main():
    reg = registration()
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    submit = json.loads((PRIVATE/'submission_pilot.json').read_text())
    job = json.loads(submit['result']['stdout'])['job_id']
    assert job == '37575963'
    record = PRIVATE/'pilot_walltime_amendment.json'
    assert not record.exists(), 'Inspect prior amendment; never repeat an uncertain mutation'
    parent = PUBLIC.parent/'european_easy_hurdle_v1/create_pilot_result.json'
    prior = json.loads(parent.read_text())
    assert prior['updates_per_head'] == 100 and prior['total_model_updates'] == 200
    assert prior['scheduler_state'] == 'COMPLETED|0:0' and prior['receipt']['seconds'] < 30
    code = r'''
import json,pathlib,re,subprocess,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['home']);job=p['job']
assert json.loads((root/'.owner.json').read_text())=={'project':'M3W','experiment':'european_easy_risk_priority_v1'}
assert json.loads((root/'submit_receipt_pilot.json').read_text())['job_id']==job
def query():
    r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr
    return r.stdout,dict(re.findall(r'(\w+)=([^\s]*)',r.stdout))
before,fields=query()
assert fields['JobId']==job and fields['JobName']=='m3w_risk_priority_pilot'
assert pathlib.Path(fields['WorkDir']).resolve()==root.resolve()
assert fields['UserId'].startswith('k24101830(') and fields['Account']=='kcl'
assert fields['Partition']=='cpu' and fields['NumCPUs']=='4' and fields['MinMemoryNode']=='16G'
assert fields['JobState']=='PENDING' and fields['RunTime']=='00:00:00' and fields['Restarts']=='0'
assert fields['TimeLimit']=='02:00:00'
intent=root/'pilot_walltime_amendment_intent.json';receipt=root/'pilot_walltime_amendment.json'
assert not intent.exists() and not receipt.exists()
intent.write_text(json.dumps({'job':job,'before':before,'requested_walltime':'00:15:00','scientific_scope_changed':False})+'\n')
r=subprocess.run(['scontrol','update','JobId='+job,'TimeLimit=00:15:00'],capture_output=True,text=True,timeout=20)
after,current=query()
out=dict(job=job,before=before,after=after,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
    walltime_verified=current['TimeLimit']=='00:15:00',new_submission=False,scope_changed=False,
    parameters_changed=False,updates_per_head=100,full_training_walltime_unchanged=True)
receipt.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''
    result = call_remote(code, dict(home=REMOTE, job=job))
    immutable(record, dict(utc=datetime.now(timezone.utc).isoformat(), result=result,
                          prior_pilot_sha256=digest(parent), code_sha256=digest(Path(__file__))))
    if result['returncode'] != 0:
        print(json.dumps(dict(returncode=result['returncode'], outcome='inspect_same_job_no_resubmit',
                              stderr_tail=result.get('stderr','')[-600:]))); return
    out = json.loads(result['stdout'])
    assert out['returncode'] == 0 and out['walltime_verified']
    immutable(PUBLIC/'pilot_walltime_amendment.json', dict(result_source='fresh_scheduler_mutation_verified',
        job=job, previous_walltime='02:00:00', corrected_pilot_walltime='00:15:00',
        prior_real_pilot_seconds=prior['receipt']['seconds'], prior_real_pilot_sha256=digest(parent),
        private_receipt_sha256=digest(record), code_sha256=digest(Path(__file__)),
        pending_only=True, new_submission=False, CPU_memory_unchanged=True,
        model_data_updates_unchanged=True, full_training_walltime='02:00:00',
        independent_roles_read=False, deployment_changed=False))
    print(json.dumps(dict(job=job, walltime_verified=True, new_submission=False, scope_changed=False)))


if __name__ == '__main__':
    main()
