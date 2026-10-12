"""Independent read-only scheduler and artifact observations, without job retries."""
import inspect
import json
import subprocess
from datetime import datetime, timezone

from scripts import manage_m3w_easy_harm_deviance as manager


def transport_failure(error):
    message=str(error)
    approval_closed=(isinstance(error,RuntimeError)
        and 'portal.er.kcl.ac.uk/mfa/' in message and 'Connection closed' in message)
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(),
        transport_observation=('ssh_access_approval_connection_closed' if approval_closed
            else 'transport_timeout_unknown' if isinstance(error,subprocess.TimeoutExpired)
            else 'transport_error_unknown'),
        scheduler_state='unknown',artifacts_verified=False,remote_modified=False,
        jobs_resubmitted=False,job_failure_inferred_from_timeout=False,
        error_type=type(error).__name__,raw_transport_error_omitted=True,
        earlier_observations_not_promoted_to_current=True)


def query(command):
    if command[0] not in ('squeue','sacct') or '-j' not in command:
        raise ValueError('Owned-job read-only query required')
    jobs=set(command[command.index('-j')+1].split(','))
    if not jobs or not jobs <= {'37835856','37835859','37837636'}:
        raise ValueError('Unexpected job scope')
    try:
        value=subprocess.run(command,capture_output=True,text=True,timeout=60)
        return dict(observation='returned',returncode=value.returncode,
                    stdout=value.stdout,stderr=value.stderr[-2000:])
    except subprocess.TimeoutExpired:
        return dict(observation='timeout_job_state_unknown',returncode=None,
                    stdout=None,stderr='Read-only scheduler query exceeded60 seconds')
    except OSError as error:
        return dict(observation='query_unavailable_job_state_unknown',returncode=None,
                    stdout=None,stderr=str(error))


REMOTE = r'''
import datetime,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
assert root==pathlib.Path('/users/k24101830/m3w/european_easy_harm_deviance_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']==root.name
for filename,job in [('train_submission.json','37835856'),('join_submission.json','37835859'),
                     ('node_portability_v1_submission.json','37837636')]:
 assert json.loads((root/filename).read_text())['job_id']==job,'Owned submission changed'
ids='37835856,37835859,37837636'
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
out=dict(started_utc=started,queue=query(['squeue','-j',ids,'-h','-o','%i|%T|%M|%R']),
 accounting=query(['sacct','-j',ids,'-X','--noheader','--parsable2',
                   '--format=JobID,State,Elapsed,ExitCode']),artifacts={},heartbeats={},
 remote_modified=False,jobs_resubmitted=False,job_failure_inferred_from_timeout=False)
public=root/'outputs/publication_readiness_2026_09'/root.name
for name in ('training_freeze.json','node_portability_v1/complete.json'):
 path=public/name
 if path.exists():
  assert path.stat().st_size<2*2**20
  raw=path.read_bytes();record=json.loads(raw)
  detail={k:v for k,v in record.items() if k not in ('fits','results')}
  out['artifacts'][name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
      fit_receipts=len(record.get('fits',[])),reference_results=len(record.get('results',[])),
      content=detail,observed_file_only_not_scheduler_confirmation=True)
 else:out['artifacts'][name]=None
private=root/'data/stage_cvpr2027_experiments'/root.name
for name in ('shard0/heartbeat.json','node_portability_v1/heartbeat.json'):
 path=private/name
 if path.exists():
  assert path.stat().st_size<1024*1024
  out['heartbeats'][name]=dict(last_written=json.loads(path.read_text()),
      file_is_not_live_process_evidence=True)
 else:out['heartbeats'][name]=None
out['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(out))
'''


def main():
    source='import subprocess\n'+inspect.getsource(query)+'\n'+REMOTE
    try:
        out=manager.base.remote(source,[manager.REMOTE],timeout=240)
    except (RuntimeError,subprocess.TimeoutExpired,OSError) as error:
        out=transport_failure(error)
    stamp=out['observed_utc'].replace('-','').replace(':','').split('.')[0]
    path=manager.PUBLIC/('resilient_observation_'+stamp+'.json')
    manager.base.once(path,out)
    print(json.dumps(dict(receipt=str(path.relative_to(manager.base.ROOT)),**out),indent=2))
    return 1 if 'transport_observation' in out else 0


if __name__=='__main__':
    raise SystemExit(main())
