"""Read immutable CREATE packets in memory; compute the registered diagnosis locally."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import select
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required')
import numpy as np
from scripts import manage_m3w_boundary_diagnostic as run
from scripts.run_m3w_boundary_diagnostic import aggregate, verify_result
from scripts.report_m3w_boundary_diagnostic import verify_summary, value
from src.world_model.m3w_boundary_diagnostic import diagnostic


SERVER=r'''
import hashlib,json,pathlib,re,sys
root=pathlib.Path(sys.argv[1]);assert root.parent==pathlib.Path('/users/k24101830/m3w')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_boundary_diagnostic_v1'
print(json.dumps({'ready':True,'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}),flush=True)
for line in sys.stdin.buffer:
    assert len(line)<=4096
    name=json.loads(line)['group'];assert re.fullmatch(r'[A-Za-z0-9_-]+',name)
    path=root/'inputs'/(name+'.npz');size=path.stat().st_size
    assert 0<size<=134217728
    raw=path.read_bytes()
    print(json.dumps(dict(group=name,bytes=size,sha256=hashlib.sha256(raw).hexdigest())),flush=True)
    sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
'''


def read_exact(stream,count,wait=False):
    if count<0 or count>128*2**20:raise ValueError('Bounded packet required')
    parts=[];remaining=count
    while remaining:
        if wait and not select.select([stream],[],[],120)[0]:
            raise TimeoutError('Input transport timed out; remote scheduler job is unaffected')
        part=stream.read(min(remaining,2**20))
        if not part:raise EOFError('Truncated immutable packet')
        parts.append(part);remaining-=len(part)
    return b''.join(parts)


def evaluate_packet(raw,entry,cfg):
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    results=[];elapsed=[]
    for repeat in range(2):
        began=time.monotonic()
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            meta=json.loads(str(z['meta_json']))
            assert meta['view']==entry['group']
            r=diagnostic({k:z[k] for k in z.files if k!='meta_json'},meta,cfg)
        r['packet_sha256']=entry['sha256'];results.append(r);elapsed.append(time.monotonic()-began)
    assert results[0]==results[1], 'Full local numerical replay differs'
    checks=verify_result(results[0])
    return results[0],checks,elapsed


def bindings():
    cfg,reg=run.registration()
    assert reg==json.loads((run.PUBLIC/'registration.json').read_text())
    paths=[Path(__file__),ROOT/'tests/test_m3w_boundary_local_stream.py',
           ROOT/'scripts/report_m3w_boundary_diagnostic.py',run.PUBLIC/'local_execution_amendment.md']
    return cfg,dict(parent_registration_sha256=run.digest(run.PUBLIC/'registration.json'),
        manifest_sha256=run.digest(run.PUBLIC/'packet_manifest.json'),
        local_code={str(p.relative_to(ROOT)):run.digest(p) for p in paths},
        registered_scientific_code={**reg['remote_code'],**reg['controls']},
        parameter_updates=0,local_row_cache_bytes=0,independent_roles_read=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');args=p.parse_args()
    cfg,reg=bindings();registration=run.PUBLIC/'local_execution_registration.json'
    if args.register:run.immutable(registration,reg);print('Registered local memory-only readout');return
    assert json.loads(registration.read_text())==reg
    run.parent.base.inter.committed(registration)
    assert not (run.PUBLIC/'local_compute_receipt.json').exists(), 'Local readout already complete'
    assert run.digest(run.HANDOFF/'compute_handoff_20260927.json')=='f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
    ssh=json.loads((run.HANDOFF/'observations.json').read_text())['ssh_arguments']
    assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
    manifest=json.loads((run.PUBLIC/'packet_manifest.json').read_text())
    command=ssh+[shlex.join(['/usr/bin/python3','-c',SERVER,run.REMOTE])]
    began=time.monotonic();records=[];refs=[];checks=0;times=np.zeros(2)
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    try:
        assert json.loads(proc.stdout.readline())==dict(ready=True,manifest_sha256=reg['manifest_sha256'])
        for entry in manifest['packets']:
            proc.stdin.write((json.dumps({'group':entry['group']})+'\n').encode());proc.stdin.flush()
            header=json.loads(proc.stdout.readline());assert header==entry
            raw=read_exact(proc.stdout,entry['bytes'],wait=True)
            result,n,seconds=evaluate_packet(raw,entry,cfg)
            del raw
            records.append(result);checks+=n;times+=seconds
            refs.append(dict(group=entry['group'],result_sha256=hashlib.sha256(
                json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()))
            if len(records)%12==0:
                beat=dict(pid=os.getpid(),groups=len(records),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          state='local_diagnosis_and_replay',local_row_cache_bytes=0)
                (run.PRIVATE/'local_heartbeat.json').write_text(json.dumps(beat)+'\n')
                print(json.dumps(beat),flush=True)
        proc.stdin.close();assert proc.wait(timeout=30)==0, proc.stderr.read().decode(errors='replace')[-1000:]
    finally:
        if proc.poll() is None:
            # Stop only this owned read-only transport, never the CREATE job.
            proc.terminate();proc.wait(timeout=15)
        proc.stdout.close();proc.stderr.close()
    assert len(records)==288
    summary=aggregate(records,cfg);summary['result_source']='fresh_run_local_memory_only_frozen_diagnostic'
    summary['accounting_checks']=checks
    summary_checks=verify_summary(summary)
    old=json.loads((run.parent.PUBLIC/'readout.json').read_text())
    prior={(r['view'],r['policy']):r['metric'] for r in old['rows']};agreements=0
    for row in summary['parent_readout_comparison']:
        for arm,scopes in row['risk'].items():
            for scope,events in scopes.items():
                policy=arm+('_matched' if scope=='matched' else '')
                for event,risk in events.items():
                    key='selected_positive_harm_ratio' if event=='all' else 'selected_easy_positive_harm_ratio'
                    run.compare_tree(risk,prior[row['view'],policy][key]);agreements+=1
    assert agreements==1728
    current_bytes=sum(p.stat().st_size for folder in (run.PUBLIC,run.PRIVATE) for p in folder.rglob('*') if p.is_file())
    assert current_bytes+len(json.dumps(summary).encode())+2**20<cfg['max_local_metadata_bytes']
    run.immutable(run.PUBLIC/'local_summary.json',summary)
    receipt=dict(groups=288,PID=os.getpid(),platform=platform.platform(),numpy=np.__version__,
        first_pass_compute_seconds=float(times[0]),replay_compute_seconds=float(times[1]),wall_seconds=time.monotonic()-began,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        full_local_replay_exact=True,native_accounting_checks=checks,summary_reduction_checks=summary_checks,
        parent_risk_agreements=agreements,result_hashes=refs,registration_sha256=run.digest(registration),
        summary_sha256=run.digest(run.PUBLIC/'local_summary.json'),remote_verification='not_run_pending_CREATE_job_37602475',
        local_row_cache_bytes=0,parameter_updates=0,independent_roles_read=False,deployment_changed=False)
    run.immutable(run.PUBLIC/'local_compute_receipt.json',receipt)
    lines=['# Local Frozen-Boundary Readout','',
        'fresh_run local diagnosis and exact numerical replay of all 288 immutable CREATE input packets. '
        'cached_verified frozen heads, actions and labels. No training or policy changes. '
        'This is not a completed CREATE run or independent confirmation.','',
        '| Phase | Arm | Event | Actual risk (%) | Predicted excess (pp) | Harm error (pp) | Reference error (pp) | Violations / defined |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for phase,row in summary['phases'].items():
        for arm in cfg['arms']:
            for event in ('all','easy'):
                z=row['arms'][arm]['matched'][event]
                vals=[value(z,k) for k in ['realized_risk_ratio','predicted_excess_over_selected_reference',
                    'harm_underestimate_over_selected_reference','reference_overestimate_over_selected_reference']]
                lines.append('| '+ ' | '.join([phase,arm,event,*vals,f"{z['violating_views']}/{z['defined_views']}"])+' |')
    lines+=['','Components share the actual selected-reference denominator. Actual risk = 2% + predicted '
        'excess + harm error + reference error. Signed components may offset. Locality means are equal-weighted. '
        'Unknown outcomes and zero denominators remain unknown, not safety passes.','',
        'The 72 fitting and 216 transfer views are dependent views of 12 previously opened localities. '
        'No fresh bootstrap, independent calibration or confirmation. Detector-silver image-local, '
        'obs8/pred12 stride12 raw frames only; no metric/seconds, true3D, foundation or human-gold claims. '
        'Stage5C and SMC remain off.','',
        f"Verified: {checks:,} native arithmetic checks; {summary_checks:,} summary checks; "
        f"{agreements:,} agreements with the parent readout. Peak RSS {receipt['peak_RSS_bytes']/2**30:.3f} GiB. "
        f"No row cache was written. Wall time {receipt['wall_seconds']:.2f}s.",'']
    (run.PUBLIC/'local_results.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in receipt.items() if k!='result_hashes'}),flush=True)


if __name__=='__main__':main()
