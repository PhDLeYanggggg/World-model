"""Recover an acknowledged-by-scheduler job, never submit a replacement job."""
import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_risk_priority import PUBLIC, PRIVATE, REMOTE, registration, digest, immutable
from scripts.manage_m3w_easy_gradient_diagnostic import call_remote


def validate_job(fields, job, home):
    assert fields['JobId'] == job and fields['JobName'] == 'm3w_risk_priority_replay'
    assert fields['UserId'].startswith('k24101830(') and fields['Account'] == 'kcl'
    assert Path(fields['WorkDir']).resolve() == Path(home).resolve()
    assert Path(fields['Command']).resolve() == (Path(home)/'run_replay.sh').resolve()
    assert fields['Partition'] == 'cpu' and fields['NumCPUs'] == '4'
    assert fields['MinMemoryNode'] == '16G' and fields['Restarts'] == '0'
    assert fields['JobState'] in ('PENDING', 'RUNNING', 'COMPLETING', 'COMPLETED')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id', required=True); args = p.parse_args()
    assert args.job_id.isdigit()
    assert registration() == json.loads((PUBLIC/'registration.json').read_text())
    failed = PRIVATE/'submission_replay.json'
    previous = json.loads(failed.read_text())
    assert 'sbatch' in previous['result']['stderr'] and 'TimeoutExpired' in previous['result']['stderr']
    record = PRIVATE/'replay_receipt_recovery.json'
    assert not record.exists(), 'Inspect the existing recovery before repeating any mutation'
    code = 'import hashlib,json,re,subprocess,sys\nfrom pathlib import Path\n'+inspect.getsource(validate_job)+r'''
p=json.loads(sys.stdin.read());root=Path(p['home']);job=p['job']
assert json.loads((root/'.owner.json').read_text())=={'project':'M3W','experiment':'european_easy_risk_priority_v1'}
intent=json.loads((root/'submit_intent_replay.json').read_text())
assert hashlib.sha256((root/'run_replay.sh').read_bytes()).hexdigest()==intent['script_sha256']
assert hashlib.sha256((root/'registration.json').read_bytes()).hexdigest()==p['registration_sha256']
trained=json.loads((root/'training_complete.json').read_text())
assert trained['groups']==108 and trained['heads']==216 and trained['control_parent_states_exact']==108
first=trained['fitting_summary'][:2]
assert len(first)==2 and all(r['step']==2000 for r in first)
assert sum(r['fitting_seconds'] for r in first)<300
a=subprocess.run(['sacct','-j',trained['job_id'],'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=20)
assert a.returncode==0 and a.stdout.strip()=='COMPLETED|0:0'
def query():
    q=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=20)
    assert q.returncode==0,q.stderr
    fields=dict(re.findall(r'(\w+)=([^\s]*)',q.stdout))
    validate_job(fields,job,root)
    return q.stdout,fields
before,fields=query()
path=root/'submit_receipt_replay.json'
assert not path.exists(), 'Do not overwrite an existing submission receipt'
receipt=dict(returncode=None,stdout='',stderr='sbatch acknowledgement timed out; existing job recovered',
    job_id=job,submission_acknowledgement_received=False,recovered_from_scheduler=True,
    script_sha256=intent['script_sha256'],recovery_code_sha256=p['code_sha256'])
with path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
changed=False;mutation=None
if fields['JobState']=='PENDING':
    assert fields['TimeLimit']=='02:00:00' and fields['RunTime']=='00:00:00'
    with (root/'replay_walltime_amendment_intent.json').open('x') as f:
        json.dump(dict(job=job,requested_walltime='00:15:00',before=before,scope_changed=False),f);f.write('\n')
    r=subprocess.run(['scontrol','update','JobId='+job,'TimeLimit=00:15:00'],capture_output=True,text=True,timeout=20)
    mutation=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    after,current=query();changed=current['TimeLimit']=='00:15:00'
    assert r.returncode==0 and changed
else:
    after,current=before,fields
out=dict(job=job,before=before,after=after,receipt=receipt,walltime_changed=changed,
    walltime=current['TimeLimit'],mutation=mutation,first_pair_fitting_seconds=sum(r['fitting_seconds'] for r in first),
    new_submission=False,CPU_memory_unchanged=True,updates_per_head=2000,model_data_unchanged=True)
with (root/'replay_receipt_recovery.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
'''
    result = call_remote(code, dict(home=REMOTE, job=args.job_id,
        registration_sha256=digest(PUBLIC/'registration.json'), code_sha256=digest(Path(__file__))))
    immutable(record, dict(utc=datetime.now(timezone.utc).isoformat(), result=result,
                          original_failed_attempt_sha256=digest(failed)))
    if result['returncode'] != 0:
        print(json.dumps(dict(status='inspect_existing_job_no_resubmit', job=args.job_id,
            stderr_tail=result.get('stderr', '')[-1000:]))); return
    out = json.loads(result['stdout'])
    immutable(PUBLIC/'replay_receipt_recovery.json', dict(result_source='fresh_scheduler_observation_and_receipt_recovery',
        job=args.job_id, prior_sbatch_acknowledgement='timed_out_not_proof_of_failure',
        recovered_from_existing_job=True, new_submission=False, model_data_updates_unchanged=True,
        walltime_changed=out['walltime_changed'], current_walltime=out['walltime'],
        first_full_pair_fitting_seconds=out['first_pair_fitting_seconds'],
        CPU_memory_unchanged=True, private_record_sha256=digest(record), code_sha256=digest(Path(__file__)),
        independent_roles_read=False, deployment_changed=False))
    print(json.dumps(dict(recovered_job=args.job_id, new_submission=False, walltime=out['walltime'])))


if __name__ == '__main__':
    main()
