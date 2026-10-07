"""One scheduling-only update to the owned, still-pending TRAIN replay."""
import inspect
import json
import subprocess

from scripts import manage_m3w_easy_harm_node_portability as diagnostic

manager = diagnostic.manager


def check_pending(fields):
    expected = dict(JobId='37837636', JobName='m3w_easy_node_portability',
        JobState='PENDING', ReqNodeList='erc-hpc-comp188', NumCPUs='4',
        MinMemoryNode='8G', TimeLimit='01:00:00', TimeMin='N/A',
        RunTime='00:00:00', Restarts='0')
    for key,value in expected.items():
        if fields.get(key) != value:
            raise ValueError('Refuse scheduling update: '+key)


def update():
    try:
        result = subprocess.run(['scontrol','update','JobId=37837636','TimeMin=00:10:00'],
                                capture_output=True,text=True,timeout=60)
        return dict(outcome='command_returned',returncode=result.returncode,
                    stdout=result.stdout,stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return dict(outcome='unknown_inspect_do_not_retry',returncode=None,
                    stdout='',stderr='Scheduler update observation timed out')


REMOTE = r'''
import datetime,json,pathlib,subprocess,sys
root=pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
expected=sys.argv[1]
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_easy_harm_recovery_v2 import verify
from scripts.train_m3w_easy_harm_deviance import sha,once,PUBLIC
verify(root)
reg=root/'node_portability_v1_registration.json'
assert sha(reg)==expected,'Diagnostic registration changed'
binding=json.loads(reg.read_text())
assert binding['node']=='erc-hpc-comp188' and binding['verification_updates']==34000
assert binding['new_scientific_fits']==0
for path,digest in binding['code_bindings'].items():
 assert sha(root/'code'/path)==digest,'Frozen diagnostic code changed'
submission=json.loads((root/'node_portability_v1_submission.json').read_text())
assert submission['job_id']=='37837636' and submission['returncode']==0
assert not (root/PUBLIC/'node_portability_v1/complete.json').exists()
intent=root/PUBLIC/'node_portability_backfill_intent_v1.json'
receipt=root/PUBLIC/'node_portability_backfill_v1.json'
assert not intent.exists() and not receipt.exists(),'Inspect prior attempt; never duplicate'
def show():
 result=subprocess.run(['scontrol','show','job','37837636','-o'],
                       capture_output=True,text=True,timeout=60)
 if result.returncode:raise RuntimeError(result.stderr)
 fields=dict(word.split('=',1) for word in result.stdout.split() if '=' in word)
 keys=('JobId','JobName','JobState','Reason','ReqNodeList','NumCPUs','MinMemoryNode',
       'TimeLimit','TimeMin','RunTime','Restarts','StartTime')
 return {key:fields.get(key) for key in keys}
before=show()
check_pending(before)
once(intent,dict(before=before,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 diagnostic_registration_sha256=expected,requested_minimum='00:10:00'))
result=update()
try:
 after=show()
except (subprocess.TimeoutExpired,RuntimeError):
 after=None
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before=before,after=after,
         **result,diagnostic_registration_sha256=expected,training_code_changed=False,
         scientific_contract_changed=False,existing_recovery_and_join_untouched=True,
         new_job_submitted=False,independent_roles_read=False)
once(receipt,out)
print(json.dumps(out))
'''


def main():
    files = [manager.base.ROOT/'scripts/allow_m3w_portability_backfill.py',
             manager.base.ROOT/'tests/test_m3w_portability_backfill.py',
             manager.PUBLIC/'node_portability_v1/backfill_protocol.md']
    for path in files:
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(manager.base.ROOT))]) != path.read_bytes():
            raise ValueError('Commit scheduling guard and protocol before mutation')
    source = ('import subprocess\n'+inspect.getsource(check_pending)+'\n'+inspect.getsource(update)+'\n'+REMOTE)
    result = manager.base.remote(source,[manager.base.sha(diagnostic.REG)],timeout=240)
    manager.base.once(manager.PUBLIC/'node_portability_backfill_v1.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
