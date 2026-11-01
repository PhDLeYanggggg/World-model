"""Read only the approved owned M3W metadata and bounded job accounting."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import manage_m3w_temporal_auxiliary_create as transport


def main():
    dest=ROOT/'data/stage_cvpr2027_experiments/european_pair_action_allocation_v1/create_owned_metadata_20261101.json'
    if dest.exists(): raise ValueError('Do not overwrite a dated observation')
    handoff=ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923/compute_handoff_20260927.json'
    if hashlib.sha256(handoff.read_bytes()).hexdigest()!='f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9':
        raise ValueError('Approved owned-path handoff changed')
    root=str(PurePosixPath(json.loads(handoff.read_text())['paths']['user_home'])/'m3w')
    code=r'''
import datetime,hashlib,json,pathlib,subprocess,sys
r=pathlib.Path(sys.argv[1]);base=r.resolve()
assert json.loads((r/'.m3w_owner.json').read_text())['project']=='M3W'
names=('training_freeze.json','producer_manifest.json','prediction_manifest.json','complete.json','runtime_receipt.json','receipt.json')
directories=[];files=[];jobs=set()
for d in sorted(r.iterdir()):
    if not d.is_dir() or d.is_symlink() or not d.resolve().is_relative_to(base):continue
    directories.append(d.name)
    for name in names:
        p=d/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size>2*1024*1024:continue
        raw=p.read_bytes()
        try:v=json.loads(raw)
        except (ValueError,UnicodeError):continue
        if not isinstance(v,dict):continue
        env=v.get('environment',{});env=env if isinstance(env,dict) else {}
        job=str(v.get('job_id',env.get('job_id','')))
        if job.isdigit():jobs.add(job)
        files.append(dict(directory=d.name,file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            modified_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),
            metadata_keys=sorted(v)[:24],job_id=job if job.isdigit() else None,
            complete_flag=v.get('complete') if isinstance(v.get('complete'),bool) else None))
ids=sorted(jobs,key=int)[-20:]
if ids:
    q=subprocess.run(['sacct','-j',','.join(ids),'-X','--noheader','--parsable2','--format=JobIDRaw,JobName,State,ExitCode'],capture_output=True,text=True,timeout=20)
    accounting=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,job_ids=ids)
else:accounting=dict(returncode=None,reason='no_job_ids_in_inspected_metadata')
print(json.dumps(dict(owned_root_verified=True,directories=directories,metadata=files,accounting=accounting,
    weights_read=False,inputs_or_labels_read=False,remote_modified=False,jobs_submitted=0,
    artifact_contents_or_science_verified=False,scope='named_small_receipts_only_no_home_or_simulation_scan')))
'''
    start=datetime.now(timezone.utc).isoformat()
    try:
        result=transport.remote(code,[root],timeout=60); observation=dict(status='read_only_metadata_observed',result=result)
    except Exception as error:
        observation=dict(status='observation_failed_not_job_failure',error_type=type(error).__name__,error=str(error))
    receipt=dict(started_utc=start,completed_utc=datetime.now(timezone.utc).isoformat(),observation=observation,
        jobs_submitted=0,remote_modified=False)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
    result=observation.get('result',{})
    print(json.dumps(dict(status=observation['status'],owned_root_verified=result.get('owned_root_verified',False),
        directories=len(result.get('directories',[])),small_receipts=len(result.get('metadata',[])),
        accounting_returncode=result.get('accounting',{}).get('returncode'),
        artifact_science_verified=False,remote_modified=False,jobs_submitted=0,
        receipt_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())))


if __name__=='__main__':main()
