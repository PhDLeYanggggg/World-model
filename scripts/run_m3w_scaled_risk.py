"""Native streamed TRAIN experiment with exact unscaled replay and safe resume."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import time

from scripts import read_m3w_task_risk_final_heads as heads
from scripts import diagnose_m3w_constraint_gradients as gradient

base=heads.manager.base; ROOT=base.ROOT
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_scaled_risk_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_scaled_risk_v1'
CONFIG=ROOT/'configs/m3w_european_scaled_risk_v1.json'; REG=PUBLIC/'registration.json'


def register():
    prior,cfg_old=heads.execution.check_registration(); cfg=json.loads(CONFIG.read_text())
    if (cfg['head_training']!=cfg_old['head_training'] or cfg['objective']!=cfg_old['objective']
            or cfg['modes']!=['unit_control','budget_scaled'] or cfg['neural_fits']!=144):
        raise ValueError('Single-factor matched scaling experiment required')
    complete=json.loads((gradient.PUBLIC/'complete.json').read_text())
    if complete['summary_sha256']!=base.sha(gradient.PUBLIC/'summary.json') or len(complete['groups'])!=216:
        raise ValueError('Complete TRAIN diagnostic required')
    for ref in complete['groups']:
        if base.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Changed diagnostic')
    files=[Path(__file__).resolve(),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_scaled_risk_training.py',
           ROOT/'tests/test_m3w_scaled_risk_training.py',ROOT/'tests/test_m3w_scaled_risk_execution.py']
    value=dict(bindings={str(p.relative_to(ROOT)):base.sha(p) for p in files},
        gradient_registration_sha256=base.sha(gradient.REG),gradient_complete_sha256=base.sha(gradient.PUBLIC/'complete.json'),
        parent_training_freeze_sha256=base.sha(heads.manager.PUBLIC/'training_freeze.json'),
        expected=prior['expected'],modes=cfg['modes'],new_fits=144,
        existing_quadratic_and_utility='cached_verified_not_fresh',independent_roles_read=False,
        threshold_search=False,deployment_changed=False)
    base.once(REG,value);return value,cfg


def admission():
    reg,cfg=register()
    for rel in [*reg['bindings'],str(REG.relative_to(ROOT))]:
        check=subprocess.run(['git','show','HEAD:'+rel],cwd=ROOT,capture_output=True)
        if check.returncode or check.stdout!=(ROOT/rel).read_bytes():raise ValueError('Commit before training')
    gradient.registration()
    return reg,cfg


def storage(cfg):
    used=sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))
    remaining=cfg['checkpoint_cap_bytes']-used;free=shutil.disk_usage(ROOT).free
    if remaining<0 or free<cfg['disk_reserve_bytes']+remaining+cfg['atomic_checkpoint_headroom_bytes']:
        raise OSError('Preserve storage reserve and checkpoint cap')
    return dict(checkpoint_bytes=used,free_bytes=free)


def execute(phase,resume):
    reg,cfg=admission()
    from src.world_model import m3w_scaled_risk_training as api
    from src.world_model.m3w_false_safe_diagnostic import moving_from_features
    from src.world_model.m3w_preprocess_portability import frozen_preprocess
    api.old.torch.set_num_threads(4);api.old.torch.set_num_interop_threads(1)
    docs,inv=heads.local_admission();parent_reg,_=heads.execution.check_registration()
    manifest=json.loads((base.PRIVATE/'create_train_input_manifest.json').read_text())
    PRIVATE.mkdir(parents=True,exist_ok=True);start=time.monotonic();refs=[];previous=None;io=0.
    finish=PUBLIC/('pilot.json' if phase=='pilot' else 'training_freeze.json')
    if finish.exists():raise ValueError('Completed phase immutable')
    if phase=='train' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real TRAIN pilot must pass')
    def beat(**kw):
        row=dict(pid=os.getpid(),phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp=PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(row,allow_nan=False)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary: resume atomic checkpoint')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Preserve atomic checkpoint')))
    with (PRIVATE/'process.lock').open('a') as lock,heads.Reader(inv) as reader:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);storage(cfg)
        for ref in (manifest['heads'][:1] if phase=='pilot' else manifest['heads']):
            identity,group=ref['identity'],ref['group']
            if group!=previous:
                beat(state='streaming_TRAIN',group=group);began=time.monotonic()
                args=heads.execution.load_packet(ref,parent_reg);io+=time.monotonic()-began;previous=group
                _,support=api.old.parent.forest.causal_inputs(args[0],args[1],args[-1])
                eligible=moving_from_features(args[0]) & support
            states=[]
            for mode in api.MODES:
                key=group+'_head'+str(identity['seed'])+'_'+mode
                path=PRIVATE/('pilot' if phase=='pilot' else 'heads')/key/'checkpoint.pt.gz'
                kw=dict(mode=mode,eligible=eligible,settings=cfg['head_training'],objective_config=cfg['objective'],
                    seed=identity['seed'],identity=identity,experiment_sha256=base.sha(REG),path=path,
                    heartbeat=beat,checkpoint_guard=lambda:storage(cfg))
                with frozen_preprocess(api.old.core,args[-1]):
                    if phase=='pilot' and mode=='budget_scaled' and not path.exists():api.fit(*args,**kw,stop_at=50)
                    state=api.fit(*args,**kw,resume=path.exists() and (resume or phase=='pilot'),
                                  stop_at=100 if phase=='pilot' else None)
                states.append(state)
                if phase=='train':
                    if mode=='unit_control':
                        doc=docs[group,identity['source'],identity['seed'],'task_risk']
                        api.assert_control(state,reader.load(ROOT/doc['checkpoint']['path']))
                    doc=dict(identity=identity,mode=mode,checkpoint_identity=state['identity'],step=state['step'],
                        trace=state['trace'],dual=state['dual'].tolist(),input_hashes=state['input_hashes'],
                        eligibility_sha256=state['eligibility_sha256'],draw_hash=state['draw_hash'],registration_sha256=base.sha(REG),
                        checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=base.sha(path),bytes=path.stat().st_size),
                        unit_control_replay_exact=mode=='unit_control',result_source='fresh_run_TRAIN_fit',
                        validation_scored=False,independent_roles_read=False)
                    p=PUBLIC/'fits'/(key+'.json');base.once(p,doc);refs.append(dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p)))
                    beat(state='fit_complete',fits=len(refs),total=144)
            api.assert_matched(states)
            if phase=='pilot':
                with frozen_preprocess(api.old.core,args[-1]):
                    direct=PRIVATE/'pilot_direct.pt.gz';orig=PRIVATE/'pilot_original.pt.gz'
                    replay=api.fit(*args,**{**kw,'path':direct},resume=resume and direct.exists(),stop_at=100)
                    control=api.old.fit(*args,**{k:v for k,v in kw.items() if k not in ('mode','path')},
                        arm='task_risk',path=orig,resume=resume and orig.exists(),stop_at=100)
                for k in replay:
                    if k!='seconds':api.old.core.exact(replay[k],states[-1][k])
                api.assert_control(states[0],control)
                estimate=(sum(s['seconds'] for s in states)*20*72+io*24)*1.5
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
                base.once(finish,dict(registration_sha256=base.sha(REG),result_source='fresh_run_real_TRAIN_pilot',
                    engineering_pass=estimate<43200 and rss<36*2**30,actual_updates=400,
                    exact_resume=True,unit_control_exact=True,matched_draws=True,peak_RSS_bytes=rss,
                    estimated_full_seconds=estimate,seconds=time.monotonic()-start,storage=storage(cfg),
                    validation_scored=False,independent_roles_read=False,scientific_lift_established=False))
                beat(state='pilot_complete',estimated_seconds=estimate)
        if phase=='train':
            if len(refs)!=144 or len({r['path'] for r in refs})!=144:raise ValueError('Full matched grid required')
            base.once(finish,dict(registration_sha256=base.sha(REG),fits=refs,neural_fits=144,source_heads=72,
                exact_previous_control_replays=72,seconds=time.monotonic()-start,io_seconds=io,
                result_source='fresh_run_complete_TRAIN_grid',validation_scored=False,independent_roles_read=False,
                deployment_changed=False,stage5c_executed=False,smc_enabled=False))
            beat(state='all144_frozen_before_readout')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','pilot','train']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='register':register();print('registered matched scaling repair')
    else:execute(a.action,a.resume)


if __name__=='__main__':main()
