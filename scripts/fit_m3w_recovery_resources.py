"""Reduce only the owned pending job's resource reservation, not its experiment."""
import inspect
import json
import re
import subprocess

from scripts import manage_m3w_easy_harm_deviance as manager


def check_pending(fields):
    expected=dict(JobId='37835856',JobName='m3w_easy_recovery_v2_train',
        JobState='PENDING',ReqNodeList='erc-hpc-comp186',NumCPUs='4',
        MinMemoryNode='16G',TimeLimit='12:00:00',TimeMin='01:00:00',
        RunTime='00:00:00',Restarts='0')
    for key,value in expected.items():
        if fields.get(key)!=value:
            raise ValueError('Refuse resource amendment: '+key)


def memory_evidence(stdout):
    expected={'37814169_1.batch','37815216_2.batch','37815216_3.batch','37817976.batch'}
    found={}
    for line in stdout.splitlines():
        values=line.strip().split('|')
        if len(values)!=5 or values[0] not in expected:
            continue
        job,state,exitcode,elapsed,rss=values
        match=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMG])',rss)
        if job in found or state!='COMPLETED' or exitcode!='0:0' or not match:
            raise ValueError('Completed unique measured batch memory required')
        size=float(match[1])*1024**('KMG'.index(match[2])+1)
        if not 0<size<=2*2**30:
            raise ValueError('Observed memory does not justify8GiB reservation')
        found[job]=dict(max_rss=rss,max_rss_bytes=size,elapsed=elapsed)
    if set(found)!=expected:
        raise ValueError('All four comparable completed memory records required')
    return found


def update():
    command=['scontrol','update','JobId=37835856','MinMemoryNode=8G','TimeMin=00:10:00']
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=90)
        return dict(outcome='command_returned',returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except subprocess.TimeoutExpired:
        return dict(outcome='unknown_inspect_do_not_retry',returncode=None,stdout='',
                    stderr='Resource update observation timed out')


REMOTE=r'''
import datetime,json,pathlib,sys
root=pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
assert json.loads((root/'train_submission.json').read_text())['job_id']=='37835856'
assert json.loads((root/'join_submission.json').read_text())['job_id']=='37835859'
sys.path.insert(0,str(root/'code'))
from scripts.train_m3w_easy_harm_recovery_v2 import verify
from scripts.train_m3w_easy_harm_deviance import sha,once,PUBLIC
verify(root)
assert sha(root/'registration.json')==sys.argv[1]
assert sha(root/'control_execution_amendment_v2.json')==sys.argv[2]
assert not (root/PUBLIC/'training_freeze.json').exists(),'Already complete; do not alter'
intent=root/PUBLIC/'scheduler_resource_repair_intent_v1.json'
receipt=root/PUBLIC/'scheduler_resource_repair_v1.json'
assert not intent.exists() and not receipt.exists(),'Inspect previous outcome, never retry blindly'
p=subprocess.run(['sacct','-j','37814169_1,37815216_2,37815216_3,37817976',
 '--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed,MaxRSS'],
 capture_output=True,text=True,timeout=60)
assert p.returncode==0
measured=memory_evidence(p.stdout)
def show():
 p=subprocess.run(['scontrol','show','job','37835856','-o'],capture_output=True,text=True,timeout=60)
 if p.returncode:raise RuntimeError(p.stderr)
 fields=dict(word.split('=',1) for word in p.stdout.split() if '=' in word)
 keys=('JobId','JobName','JobState','Reason','ReqNodeList','NumCPUs','MinMemoryNode',
       'TimeLimit','TimeMin','RunTime','Restarts','StartTime')
 return {key:fields.get(key) for key in keys}
before=show()
check_pending(before)
once(intent,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 before=before,measured_completed_batches=measured,requested_memory='8G',
 minimum_allocation='00:10:00',requested_maximum='12:00:00',
 training_registration_sha256=sys.argv[1],execution_amendment_sha256=sys.argv[2]))
result=update()
try:
 after=show()
except (subprocess.TimeoutExpired,RuntimeError):
 after=None
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before=before,after=after,
 measured_completed_batches=measured,**result,training_code_changed=False,
 scientific_contract_changed=False,node_changed=False,cpu_threads_changed=False,
 jobs_submitted=False,checkpoints_preserved=True,join_untouched=True,independent_roles_read=False)
once(receipt,out)
print(json.dumps(out))
'''


def main():
    root=manager.base.ROOT
    for path in (root/'scripts/fit_m3w_recovery_resources.py',
                 root/'tests/test_m3w_recovery_resource_fit.py',
                 manager.PUBLIC/'scheduler_resource_repair_protocol.md'):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(root))])!=path.read_bytes():
            raise ValueError('Commit guard, tests and scheduling protocol before mutation')
    source='import re,subprocess\n'+'\n'.join(inspect.getsource(f) for f in
        (check_pending,memory_evidence,update))+'\n'+REMOTE
    out=manager.base.remote(source,[manager.base.sha(manager.REG),
        manager.base.sha(manager.PUBLIC/'control_execution_amendment_v2.json')],timeout=300)
    manager.base.once(manager.PUBLIC/'scheduler_resource_repair_v1.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
