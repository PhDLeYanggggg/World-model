"""Registered shared/local risk fitting and evaluation inside source TRAIN only."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from scripts import run_m3w_train_census as census

ROOT, base = census.ROOT, census.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_shared_train_risk_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_shared_train_risk_v1'
CONFIG = ROOT/'configs/m3w_european_shared_train_risk_v1.json'
REG = PUBLIC/'registration.json'


def register():
    from src.world_model.m3w_shared_train_risk import recording_partition
    cfg = json.loads(CONFIG.read_text())
    manifest = json.loads((base.PRIVATE/'create_train_input_manifest.json').read_text())
    complete = json.loads((census.PUBLIC/'complete.json').read_text())
    if (complete['summary_sha256'] != base.sha(census.PUBLIC/'summary.json')
            or manifest['input_role'] != 'source_TRAIN_only' or manifest['validation_rows_transferred']):
        raise ValueError('Verified TRAIN-only lineage required')
    contexts = {}
    for ref in manifest['heads']:
        if ref['identity']['seed'] != 17: continue
        group = ref['group'];source=ref['identity']['source']
        path = census.PUBLIC/'groups'/(group+'_head17_budget_scaled.json')
        expected = next(r['sha256'] for r in complete['groups'] if r['path'] == str(path.relative_to(ROOT)))
        if base.sha(path) != expected: raise ValueError('Changed source recording roster')
        row=json.loads(path.read_text())
        context=group.rsplit('_fit_',1)[0]
        if context != ref['identity']['upstream']['context']:
            raise ValueError('Producer/controller context mismatch')
        contexts.setdefault(context,[]).append(dict(packet=ref,partition=recording_partition(
            source,list(row['census']['recordings'])),census_sha256=expected))
    if len(contexts)!=6 or any(len(v)!=4 for v in contexts.values()):
        raise ValueError('Complete six by four TRAIN packet design required')
    files=[Path(__file__).resolve(),CONFIG,PUBLIC/'protocol.md',
        ROOT/'src/world_model/m3w_shared_train_risk.py',ROOT/'tests/test_m3w_shared_train_risk.py']
    parent_reg,_=census.heads.previous.execution.check_registration()
    value=dict(bindings={str(p.relative_to(ROOT)):base.sha(p) for p in files},
        parent_manifest_sha256=base.sha(base.PRIVATE/'create_train_input_manifest.json'),
        parent_registration_sha256=base.sha(census.heads.previous.execution.manager.REG),
        prospective_contract_sha256=base.sha(ROOT/'outputs/publication_readiness_2026_09/european_gate_contract_v1/prospective_protocol.md'),
        contexts=contexts,seeds=cfg['head_seeds'],fits=108,outer_development_read=False,
        independent_roles_read=False,threshold_search=False,deployment_changed=False)
    base.once(REG,value)
    return value,cfg,parent_reg


def admission():
    reg,cfg,parent=register()
    for rel in [*reg['bindings'],str(REG.relative_to(ROOT))]:
        result=subprocess.run(['git','show','HEAD:'+rel],cwd=ROOT,capture_output=True)
        if result.returncode or result.stdout!=(ROOT/rel).read_bytes():
            raise ValueError('Commit protocol and code before real fitting')
    return reg,cfg,parent


def storage(cfg):
    used=sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))
    free=shutil.disk_usage(ROOT).free
    if used>cfg['checkpoint_cap_bytes'] or free<cfg['disk_reserve_bytes']+cfg['checkpoint_cap_bytes']:
        raise OSError('Preserve10GiB reserve and256MiB model cap')
    return dict(checkpoint_bytes=used,free_bytes=free)


def packets(refs,parent,beat):
    from src.world_model import m3w_shared_train_risk as api
    fit,held={},{ }
    for item in refs:
        ref=item['packet'];source=ref['identity']['source']
        beat(state='streaming_TRAIN',source=source)
        args=census.heads.previous.execution.load_packet(ref,parent)
        fit[source],held[source]=api.prepare(args,item['partition'])
    return fit,held


def path_for(context,seed,mode,source=None,pilot=False):
    key=context+'_head'+str(seed)+'_'+mode+('_'+source if source else '')
    return PRIVATE/('pilot' if pilot else 'heads')/(key+'.pt.gz')


def execute(phase,resume):
    reg,cfg,parent=admission()
    from src.world_model import m3w_shared_train_risk as api
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True);started=time.monotonic();refs=[]
    finish=PUBLIC/('pilot.json' if phase=='pilot' else 'training_freeze.json')
    if finish.exists(): raise ValueError('Completed phase immutable')
    if phase=='train' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real TRAIN/resume pilot required')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    def beat(**kw):
        row=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),phase=phase,**kw)
        tmp=PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(row,allow_nan=False)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
        if time.monotonic()-started>cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume, do not lower experiment scope')
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);storage(cfg)
        chosen=list(sorted(reg['contexts']))[:1] if phase=='pilot' else sorted(reg['contexts'])
        seeds=reg['seeds'][:1] if phase=='pilot' else reg['seeds']
        for context in chosen:
            fit,_=packets(reg['contexts'][context],parent,beat)
            for seed in seeds:
                locals={};shared={}
                common=dict(settings=cfg['head_training'],objective_config=cfg['objective'],seed=seed,
                    experiment_sha256=base.sha(REG),heartbeat=beat)
                for mode in cfg['arms']:
                    for source in (sorted(fit) if mode=='local' else [None]):
                        path=path_for(context,seed,mode,source,phase=='pilot')
                        identity=dict(context=context,seed=seed,mode=mode,source=source,
                            inner_partitions={r['packet']['identity']['source']:r['partition'] for r in reg['contexts'][context]})
                        kw=dict(**common,identity=identity,path=path,resume=resume and path.exists(),
                            stop_at=100 if phase=='pilot' else None)
                        if mode=='local':
                            state=api.local.fit(*fit[source],mode='budget_scaled',eligible=api.eligible(fit[source]),
                                checkpoint_guard=lambda:storage(cfg),**kw)
                            locals[source]=state
                        else:
                            if phase=='pilot' and mode=='shared_source' and not path.exists():
                                api.fit_shared(fit,mode=mode,guard=lambda:storage(cfg),**{**kw,'stop_at':50})
                                kw['resume']=True
                            state=api.fit_shared(fit,mode=mode,guard=lambda:storage(cfg),**kw)
                            shared[mode]=state
                        if phase=='train':
                            doc=dict(identity=identity,checkpoint=dict(path=str(path.relative_to(ROOT)),
                                sha256=base.sha(path),bytes=path.stat().st_size),step=state['step'],
                                registration_sha256=base.sha(REG),result_source='fresh_run_inner_TRAIN_fit',
                                inner_held_scored=False,outer_development_read=False,independent_roles_read=False,
                                final_trace=state['trace'][-1])
                            receipt=PUBLIC/'fits'/(path.stem.replace('.pt','')+'.json');base.once(receipt,doc)
                            refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=base.sha(receipt)))
                            beat(state='fit_complete',fits=len(refs),total=108)
                for name in ('preprocess','source_preprocess','initial_model','input_hashes','samplers'):
                    api.old.core.exact(shared['shared_global'][name],shared['shared_source'][name])
                for source,state in locals.items():
                    sampling=shared['shared_source']['samplers'][source]
                    api.old.core.exact(sampling['rng'],state['sampler_rng'])
                    if sampling['draw_hash']!=state['draw_hash'] or sampling['row_draws']!=state['row_draws']:
                        raise ValueError('Per-source training exposure differs')
                if phase=='pilot':
                    direct=PRIVATE/'pilot_direct.pt.gz'
                    replay=api.fit_shared(fit,mode='shared_source',guard=lambda:storage(cfg),
                        **{**kw,'path':direct,'resume':resume and direct.exists()})
                    for name in replay:
                        if name!='seconds':api.old.core.exact(replay[name],shared['shared_source'][name])
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    elapsed=time.monotonic()-started
                    estimate=elapsed*20*18*1.5
                    base.once(finish,dict(seconds=elapsed,peak_RSS_bytes=rss,estimated_full_seconds=estimate,
                        engineering_pass=estimate<43200 and rss<36*2**30,exact_resume=True,
                        matched_per_source_draws=True,actual_updates=700,inner_held_scored=False,
                        outer_development_read=False,independent_roles_read=False,
                        registration_sha256=base.sha(REG),storage=storage(cfg)))
                    beat(state='pilot_complete',estimated_full_seconds=estimate)
        if phase=='train':
            if len(refs)!=108 or len({r['path'] for r in refs})!=108:raise ValueError('All108 required')
            base.once(finish,dict(fits=refs,registration_sha256=base.sha(REG),seconds=time.monotonic()-started,
                fresh_fits=108,optimizer_steps=216000,matched_per_source_draws=True,inner_held_scored=False,
                outer_development_read=False,independent_roles_read=False,deployment_changed=False))
            beat(state='all108_frozen_before_inner_readout')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','pilot','train'])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='register':register();print('Registered108 TRAIN-internal fits')
    else:execute(a.action,a.resume)


if __name__=='__main__': main()
