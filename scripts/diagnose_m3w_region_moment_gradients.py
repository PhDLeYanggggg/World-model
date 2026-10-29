"""Registered TRAIN-only moment-gradient diagnosis of54frozen current heads."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_policy_region_cost as parent
from src.evaluation.m3w_region_gradient_diagnosis import sampling, diagnose

PUBLIC = parent.PUBLIC.parent/'european_region_gradient_v1'
PRIVATE = parent.PRIVATE.parent/'european_region_gradient_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), ROOT/'src/world_model/m3w_exact_region_cost.py',
    ROOT/'src/evaluation/m3w_region_gradient_diagnosis.py', ROOT/'tests/test_m3w_exact_region_cost.py',
    PUBLIC/'protocol.md')


def registration():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    parent.once(REG,dict(bindings={str(p.relative_to(ROOT)):parent.sha(p) for p in FILES},
        parent_training_freeze_sha256=parent.sha(parent.PUBLIC/'training_freeze.json'),
        parent_verification_sha256=parent.sha(parent.PUBLIC/'verification.json'),
        states=54, probes_per_state=64, sampler_steps=2000, include_held=False,
        optimizer_updates=0, independent_roles_opened=False))


def run(verify=False):
    cfg, pc, reg, source, jobs, identity, teachers, packed, polygons = parent.admission()
    binding=json.loads(REG.read_text())
    if (binding['parent_training_freeze_sha256']!=parent.sha(parent.PUBLIC/'training_freeze.json')
            or binding['parent_verification_sha256']!=parent.sha(parent.PUBLIC/'verification.json')):
        raise ValueError('The registered frozen endpoint collection changed')
    for rel,digest in binding['bindings'].items():
        if parent.sha(ROOT/rel)!=digest: raise ValueError('Frozen gradient diagnosis changed')
        parent.parent.committed(ROOT/rel)
    parent.parent.committed(REG); models=parent.models()
    if (PUBLIC/'complete.json').exists() and not verify:
        raise ValueError('Completed diagnosis immutable; use verify')
    parent.torch.set_num_threads(4); parent.torch.set_num_interop_threads(1)
    started=time.monotonic()
    def beat(**event):
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiBdisk reserve')
        if time.monotonic()-started>cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12hhealthy-runtime boundary')
        value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        parent.json_write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    outputs=[]
    for context,items in sorted(reg['contexts'].items()):
        packets=parent.old.scene_packets(parent.parent.prepare_context(context,items,source,jobs,identity,beat,include_held=False),
            source,packed,polygons)['fit']; data=parent.old.training_data(packets)
        data['diagnostic_settings']=cfg['training']
        for seed in cfg['seeds']:
            for arm in parent.parent.api.ARMS:
                ref=models[context,seed,arm]; state=parent.parent.api.read_checkpoint(ROOT/ref['checkpoint']['path'])
                regions=parent.region_data(packets,teachers[context,seed,arm]['checkpoint'],arm)
                for key in ('pool','local_action','target'):
                    if hashlib.sha256(regions[key].numpy().tobytes()).hexdigest()!=state['region_arrays'][key]:
                        raise ValueError('Unchanged fitting-region arrays required')
                draws=sampling(data,regions,cfg['training'],seed)
                if draws['draw_hash']!=state['draw_hash']: raise ValueError('Original2000batch draws differ')
                model=parent.api.SceneCostHead(state['dimension'],state['settings']['width'],seed,state['preprocess']['bias'])
                model.load_state_dict(state['model']); model.eval()
                row=diagnose(model,data,regions,parent.torch.tensor(parent.parent.api.ARMS[arm]),
                    parent.torch.from_numpy(state['preprocess']['rms']),draws.pop('batches'))
                value=dict(context=context,seed=seed,arm=arm,diagnosis=row,sampling=draws,
                    checkpoint_sha256=ref['checkpoint']['sha256'],registration_sha256=parent.sha(REG))
                path=PUBLIC/'groups'/f'{context}_seed{seed}_{arm}.json'
                if verify:
                    if value!=json.loads(path.read_text()): raise ValueError('Exact gradient replay differs')
                else: parent.once(path,value)
                outputs.append(dict(path=str(path.relative_to(ROOT)),sha256=parent.sha(path)))
                beat(state='region_gradient_progress',states=len(outputs),total=54,context=context,seed=seed,arm=arm)
        del data,packets
    rows=[json.loads((ROOT/r['path']).read_text()) for r in outputs]; summary={}
    for arm in parent.parent.api.ARMS:
        subset=[r for r in rows if r['arm']==arm]
        median=lambda key:statistics.median(r['diagnosis'][key] for r in subset if r['diagnosis'][key] is not None)
        summary[arm]=dict(states=len(subset),median_minibatch_vs_exact_cosine=median('minibatch_mean_vs_exact_cosine'),
            median_minibatch_relative_error=median('minibatch_mean_relative_error'),
            median_base_vs_EH_cosine=median('base_vs_EH_cosine'),
            median_other_moments_vs_EH_cosine=median('other_moments_vs_EH_cosine'),
            empty_auxiliary_steps=sum(r['sampling']['zero_region_steps'] for r in subset),
            total_original_steps=2000*len(subset))
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()) or outputs!=json.loads((PUBLIC/'complete.json').read_text())['groups']:
            raise ValueError('Exact54-state gradient summary differs')
        parent.once(PUBLIC/'verification.json',dict(exact_replay=True,states=54,summary_sha256=parent.sha(PUBLIC/'summary.json')))
    else:
        parent.once(PUBLIC/'summary.json',summary)
        parent.once(PUBLIC/'complete.json',dict(pid=os.getpid(),seconds=time.monotonic()-started,groups=outputs,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,optimizer_updates=0,
            result_source='fresh_frozen_TRAIN_gradient_diagnosis',independent_roles_opened=False))
    parent.json_write(PRIVATE/('verification_runtime.json' if verify else 'runtime.json'),dict(pid=os.getpid(),seconds=time.monotonic()-started))
    beat(state='region_gradient_verified' if verify else 'region_gradient_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','run','verify'));a=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    if a.phase=='register':registration()
    else:
        with (PRIVATE/'process.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a.phase=='verify')
