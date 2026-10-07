"""Permit bounded backfill for the existing owned pending recovery, no resubmit."""
import json
import subprocess

from scripts import manage_m3w_easy_harm_deviance as manager

REMOTE=r'''
import datetime,json,pathlib,subprocess,sys
r=pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((r/'.owner.json').read_text())['experiment']==r.name
assert json.loads((r/'train_submission.json').read_text())['job_id']=='37835856'
assert json.loads((r/'join_submission.json').read_text())['job_id']=='37835859'
sys.path.insert(0,str(r/'code'))
from scripts.train_m3w_easy_harm_recovery_v2 import verify
from scripts.train_m3w_easy_harm_deviance import once,PUBLIC,sha
verify(r)
receipt=r/PUBLIC/'backfill_time_min_v1.json'
assert not receipt.exists(),'Existing update: inspect, do not repeat'
def show():
 p=subprocess.run(['scontrol','show','job','37835856','-o'],capture_output=True,text=True,timeout=20)
 assert p.returncode==0
 fields=dict(s.split('=',1) for s in p.stdout.split() if '=' in s)
 keys=('JobId','ArrayJobId','ArrayTaskId','JobName','JobState','Reason','TimeLimit','TimeMin','RunTime','ReqNodeList','NumCPUs','MinMemoryNode','Restarts','StartTime')
 return {k:fields.get(k) for k in keys}
before=show()
assert before['JobState']=='PENDING' and before['JobName']=='m3w_easy_recovery_v2_train'
assert before['ReqNodeList']=='erc-hpc-comp186' and before['TimeLimit']=='12:00:00'
assert before['TimeMin']=='N/A' and before['RunTime']=='00:00:00' and before['NumCPUs']=='4'
intent=r/PUBLIC/'backfill_time_min_intent_v2.json'
assert not intent.exists(),'Inspect existing intent; never repeat an ambiguous update'
once(intent,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before=before,
 previous_attempt='20-second timeout; later read-only observation confirmed TimeMin=N/A'))
def update():
 try:
  p=subprocess.run(['scontrol','update','JobId=37835856','TimeMin=01:00:00'],capture_output=True,text=True,timeout=60)
  return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,outcome='command_returned')
 except subprocess.TimeoutExpired:
  return dict(returncode=None,stdout='',stderr='scontrol update timed out after60 seconds',outcome='unknown_inspect_do_not_retry')
result=update()
try:
 after=show()
except (subprocess.TimeoutExpired,AssertionError):
 after=None
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before=before,after=after,
 **result,
 registration_sha256=sha(r/'registration.json'),execution_amendment_sha256=sha(r/'control_execution_amendment_v2.json'),
 training_code_changed=False,scientific_contract_changed=False,new_jobs_submitted=False,
 checkpoint_resume_required_if_time_limited=True,independent_roles_read=False)
once(receipt,out)
print(json.dumps(out))
'''


def main():
    path=manager.base.ROOT/'scripts/allow_m3w_recovery_backfill.py'
    if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(manager.base.ROOT))])!=path.read_bytes():
        raise ValueError('Commit scheduler amendment before update')
    result=manager.base.remote(REMOTE,[],timeout=150)
    manager.base.once(manager.PUBLIC/'backfill_time_min_v1.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
