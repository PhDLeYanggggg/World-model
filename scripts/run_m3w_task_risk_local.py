"""Native local execution of the registered objective, streaming TRAIN packets."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import signal
import subprocess
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')

from scripts import manage_m3w_task_risk as manager

base, PUBLIC = manager.base, manager.PUBLIC
PRIVATE = base.ROOT/'data/stage_cvpr2027_experiments/european_task_risk_v1/local'
AMENDMENT = PUBLIC/'local_execution_amendment.json'

PACKET = r'''
import hashlib,json,pathlib,re,sys
root=pathlib.Path('/users/k24101830/m3w/european_temporal_auxiliary_v1')
group,manifest_hash,parent_hash=sys.argv[1:]
assert re.fullmatch(r'[A-Za-z0-9_-]+',group)
assert json.loads((root/'.owner.json').read_text())['registration_sha256']==parent_hash
raw=(root/'train_input_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest_hash
m=json.loads(raw);assert m['input_role']=='source_TRAIN_only' and not m['validation_rows_transferred'] and not m['independent_roles_read']
refs=[r for r in m['heads'] if r['group']==group];assert len(refs)==3
path=root/'inputs'/(group+'.npz');assert path.resolve().is_relative_to((root/'inputs').resolve())
assert path.stat().st_size==refs[0]['bytes']
with path.open('rb') as f:
 while True:
  b=f.read(1024*1024)
  if not b:break
  sys.stdout.buffer.write(b)
'''


def register():
    manager.register()
    files = [Path(__file__).resolve(), base.ROOT/'tests/test_m3w_task_risk_local.py',
             PUBLIC/'local_execution.md']
    value = dict(training_registration_sha256=base.sha(manager.REG),
        bindings={str(p.relative_to(base.ROOT)):base.sha(p) for p in files},
        create_quota_limit_bytes=50000000000, create_quota_used_bytes=45572776137,
        create_free_bytes=4427223863, registered_reserve_bytes=10737418240,
        create_submissions=0, execution='native_arm64_local_CPU_streamed_remote_TRAIN',
        numerical_core_changed=False, original_optimizer_replay_required=True,
        same_environment_matching_required=True, historical_cross_platform_identity_claim=False,
        packet_disk_cache=False, full_fits=216, steps=2000, risk_budget=.02,
        independent_roles_read=False, threshold_search=False, deployment_changed=False)
    base.once(AMENDMENT,value)
    return value


def check_registration():
    amend=register(); reg=manager.register()
    for rel in [*reg['bindings'],*amend['bindings'],str(manager.REG.relative_to(base.ROOT)),
                str(AMENDMENT.relative_to(base.ROOT))]:
        p=subprocess.run(['git','show','HEAD:'+rel],cwd=base.ROOT,capture_output=True)
        if p.returncode or p.stdout!=(base.ROOT/rel).read_bytes():
            raise ValueError('Commit registered local execution before numerical work')
    return reg,json.loads(manager.CONFIG.read_text())


def load_packet(ref,reg):
    args=[ref['group'],reg['input_manifest_sha256'],reg['parent_registration_sha256']]
    p=subprocess.run(base.ssh_args()+[shlex.join(['/usr/bin/python3','-c',PACKET,*args])],
                     capture_output=True,timeout=300)
    if p.returncode:
        raise RuntimeError('TRAIN stream failed; no resubmission or authentication changes: '+str(p.returncode))
    if len(p.stdout)!=ref['bytes'] or hashlib.sha256(p.stdout).hexdigest()!=ref['sha256']:
        raise ValueError('TRAIN packet byte/hash mismatch')
    from scripts.train_m3w_temporal_auxiliary_portable import unpack
    return unpack(p.stdout)


def storage(cfg):
    used=sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))
    remaining=cfg['checkpoint_cap_bytes']-used
    free=shutil.disk_usage(base.ROOT).free
    if remaining<0 or free<cfg['disk_reserve_bytes']+remaining+cfg['atomic_checkpoint_headroom_bytes']:
        raise OSError('Preserve 10GiB local reserve and registered checkpoint cap')
    return dict(free_bytes=free,checkpoint_bytes=used)


def execute(phase,resume=False):
    reg,cfg=check_registration()
    from src.world_model import m3w_task_risk_training as api
    from src.world_model.m3w_false_safe_diagnostic import moving_from_features
    from src.world_model.m3w_preprocess_portability import frozen_preprocess
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    finish=PUBLIC/('local_pilot.json' if phase=='pilot' else 'training_freeze.json')
    if finish.exists():raise ValueError('Completed phase is immutable')
    if phase=='train' and not json.loads((PUBLIC/'local_pilot.json').read_text())['engineering_pass']:
        raise ValueError('Successful real TRAIN pilot required')
    manifest=json.loads((base.PRIVATE/'create_train_input_manifest.json').read_text())
    movements={r['group']:r for r in reg['movement']['groups']}
    started=time.monotonic(); registration=base.sha(manager.REG); amendment=base.sha(AMENDMENT)

    def beat(**kw):
        row=dict(pid=os.getpid(),phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp=PRIVATE/(phase+'_heartbeat.tmp');tmp.write_text(json.dumps(row,allow_nan=False)+'\n')
        os.replace(tmp,PRIVATE/(phase+'_heartbeat.json'))
        with (PRIVATE/(phase+'_events.jsonl')).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
        if time.monotonic()-started>cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12-hour boundary; preserve checkpoint and resume, do not shrink target')

    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        storage(cfg);previous=None;refs=[];io_seconds=0.
        for ref in (manifest['heads'][:1] if phase=='pilot' else manifest['heads']):
            identity,group=ref['identity'],ref['group']
            if group!=previous:
                beat(state='streaming_verified_TRAIN_packet',group=group)
                began=time.monotonic();args=load_packet(ref,reg);io_seconds+=time.monotonic()-began
                moving=moving_from_features(args[0]);_,support=api.parent.forest.causal_inputs(args[0],args[1],args[-1])
                eligible=moving & support
                if (api.parent.forest.fingerprint(moving)!=movements[group]['moving_sha256']
                        or ref['sha256']!=movements[group]['packet_sha256']):
                    raise ValueError('Causal eligibility lineage changed')
                previous=group
            states=[]
            for arm in api.ARMS:
                key=group+'_head'+str(identity['seed'])+'_'+arm
                path=PRIVATE/('pilot_heads' if phase=='pilot' else 'heads')/key/'checkpoint.pt.gz'
                kw=dict(eligible=eligible,arm=arm,settings=cfg['head_training'],objective_config=cfg['objective'],
                    seed=identity['seed'],identity=identity,experiment_sha256=registration,path=path,
                    heartbeat=beat,checkpoint_guard=lambda:storage(cfg))
                with frozen_preprocess(api.core,args[-1]):
                    if phase=='pilot' and arm=='task_risk' and not path.exists():api.fit(*args,**kw,stop_at=50)
                    s=api.fit(*args,**kw,resume=path.exists() and (resume or phase=='pilot'),
                              stop_at=100 if phase=='pilot' else None)
                states.append(s)
                if phase=='train':
                    doc=dict(identity=identity,arm=arm,step=s['step'],trace=s['trace'],dual=s['dual'].tolist(),
                        input_hashes=s['input_hashes'],eligibility_sha256=s['eligibility_sha256'],draw_hash=s['draw_hash'],
                        registration_sha256=registration,local_amendment_sha256=amendment,
                        checkpoint=dict(path=str(path.relative_to(base.ROOT)),sha256=base.sha(path),bytes=path.stat().st_size),
                        validation_scored=False,independent_roles_read=False,result_source='fresh_run_native_TRAIN_fit')
                    output=PUBLIC/'fits'/(key+'.json');base.once(output,doc)
                    refs.append(dict(path=str(output.relative_to(base.ROOT)),sha256=base.sha(output)))
                    beat(state='fit_complete',fits=len(refs),total=216,key=key)
            api.assert_matched(states)
            if phase=='pilot':
                with frozen_preprocess(api.core,args[-1]):
                    direct=PRIVATE/'pilot_direct.pt.gz'
                    replay=api.fit(*args,**{**kw,'path':direct},resume=resume and direct.exists(),stop_at=100)
                    cp=PRIVATE/'pilot_original.pt.gz'
                    control=api.parent.fit(*args,arm='none',settings=cfg['head_training'],seed=identity['seed'],
                        identity=identity,path=cp,heartbeat=beat,resume=resume and cp.exists(),stop_at=100,
                        checkpoint_guard=lambda:storage(cfg))
                for k in replay:
                    if k!='seconds':api.core.exact(replay[k],states[-1][k])
                api.assert_original_control(states[0],control)
                estimate=(sum(s['seconds'] for s in states)/100*2000*72+io_seconds*24)*1.5
                rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                if platform.system()!='Darwin':rss*=1024
                base.once(finish,dict(result_source='fresh_run_real_TRAIN_local_pilot',identity=identity,
                    registration_sha256=registration,local_amendment_sha256=amendment,pid=os.getpid(),
                    architecture=platform.machine(),torch=api.torch.__version__,numpy=api.np.__version__,
                    cpu_threads=4,num_workers=0,updates_per_arm=100,actual_updates_including_replays=500,
                    exact_interrupted_resume=True,original_control_exact=True,matched_draws=True,
                    peak_RSS_bytes=rss,estimated_full_seconds_with_1_5_margin=estimate,
                    engineering_pass=estimate<43200 and rss<36*2**30,io_seconds=io_seconds,
                    seconds=time.monotonic()-started,storage=storage(cfg),
                    monitor={s['arm']:dict(initial=s['trace'][0],final=s['trace'][-1]) for s in states},
                    scientific_lift_established=False,validation_scored=False,independent_roles_read=False,
                    deployment_changed=False))
                beat(state='pilot_complete',estimated_full_seconds=estimate)
        if phase=='train':
            if len(refs)!=216 or len({r['path'] for r in refs})!=216:raise ValueError('Full fixed grid required')
            base.once(finish,dict(result_source='fresh_run_native_TRAIN_fit_grid',fits=refs,neural_fits=216,
                source_heads=72,registration_sha256=registration,local_amendment_sha256=amendment,
                seconds=time.monotonic()-started,io_seconds=io_seconds,validation_scored=False,
                independent_roles_read=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
            beat(state='all216_frozen_before_readout')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','pilot','train'])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='register':register();print(json.dumps(dict(amendment_sha256=base.sha(AMENDMENT))))
    else:execute(a.mode,a.resume)


if __name__=='__main__':main()
