"""Registered same-record/cross-record meta-training inside original TRAIN."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time

from scripts import run_m3w_whole_record_policy as previous
from src.world_model import m3w_cross_record_policy as api
from src.evaluation import m3w_cross_record_readout as readout

ROOT, sha, once = previous.ROOT, previous.sha, previous.once
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cross_record_policy_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_cross_record_policy_v1'
CONFIG = ROOT/'configs/m3w_cross_record_policy_v1.json'
REG = PUBLIC/'registration.json'


def register():
    reg, old, ancestor = previous.admission(True); cfg = json.loads(CONFIG.read_text())
    extra = ('inner_learning_rate','inner_gradient_clip','meta_weight')
    if ({k:v for k,v in cfg['training'].items() if k not in extra} != old['training']
            or cfg['head_seeds'] != reg['seeds'] or cfg['risk_budget'] != .02):
        raise ValueError('Preserve architecture, update budget, features and risk budget')
    support = {}
    for items in reg['contexts'].values():
        for item in items:
            source = item['packet']['identity']['source']; part = item['partition']
            record = {k:part[k] for k in ('fit_recordings','held_recordings')}
            if source in support and support[source] != record: raise ValueError('Split disagreement')
            support[source] = record
    paths = [Path(__file__).resolve(), CONFIG, PUBLIC/'protocol.md',
        ROOT/'src/world_model/m3w_cross_record_policy.py', ROOT/'src/evaluation/m3w_cross_record_readout.py',
        ROOT/'tests/test_m3w_cross_record_policy.py', ROOT/'tests/test_m3w_cross_record_readout.py',
        previous.REG, previous.PUBLIC/'training_freeze.json', previous.PUBLIC/'complete.json',
        previous.PUBLIC/'summary.json']
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], primary=cfg['primary'], arms=cfg['arms'],
        fits=36, steps_per_fit=2000, expected_views_per_scope=72, inference_threshold=0.,
        source_recording_support=support,
        single_fit_recording_sources=sorted(s for s,p in support.items() if len(p['fit_recordings']) == 1),
        official_split_changed=False, episode_sides_both_training=True, test_time_adaptation=False,
        independent_roles_read=False, inner_held_reused_exploratory=True,
        whole_pipeline_OOF=False, deployment_changed=False))
    return reg, cfg, ancestor


def admission():
    out = register()
    for rel in [*json.loads(REG.read_text())['bindings'], str(REG.relative_to(ROOT))]:
        previous.previous.previous.pooled.previous.previous.committed(ROOT/rel)
    return out


def runtime(cfg):
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); began = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve required')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Registered private checkpoint cap exceeded')
    def beat(**kw):
        event = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        tmp = PRIVATE/'heartbeat.tmp'; tmp.write_text(json.dumps(event)+'\n'); os.replace(tmp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps({k:v for k,v in event.items() if k not in ('dual','exact_record_risk')}), flush=True)
        if time.monotonic()-began > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h limit; resume same scope')
    guard(); beat(state='admitted'); return began, beat, guard


def path_for(context, seed, arm, pilot=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}.pt.gz'


def train(pilot=False, resume=False):
    reg, cfg, ancestor = admission(); _, origins = previous.previous.ORIGINS.models(reg)
    parents = previous.previous.models(); finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists(): raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real pilot required')
    began, beat, guard = runtime(cfg); receipts, pilot_states = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        contexts = sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
        for context, items in contexts:
            packets, held = previous.previous.LOADER.packets(items, ancestor, beat); del held
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                data, parent, origin_ref, parent_ref = previous.prepare(context, seed, packets, origins, parents)
                episode_hashes = []
                for arm in api.ARMS:
                    path = path_for(context, seed, arm, pilot)
                    kw = dict(data=data, arm=arm, settings=cfg['training'], seed=seed,
                        identity=dict(context=context, seed=seed, registration_sha256=sha(REG),
                            origin_sha256=origin_ref['sha256'], parent_sha256=parent_ref['sha256']),
                        path=path, heartbeat=beat, guard=guard)
                    if pilot and not path.exists(): api.fit(**kw, stop_at=10)
                    state = api.fit(**kw, stop_at=20 if pilot else None, resume=path.exists() and (resume or pilot))
                    api.base.core.exact(state['initial_model'], parent['initial_model'])
                    if not any(not api.torch.equal(v, state['initial_model'][k]) for k,v in state['model'].items()):
                        raise ValueError('No real parameter update')
                    episode_hashes.append(state['episode_hash'])
                    if pilot:
                        direct_path = PRIVATE/'pilot'/f'{arm}_direct.pt.gz'
                        direct = api.fit(**{**kw,'path':direct_path}, stop_at=20, resume=resume and direct_path.exists())
                        for k in direct:
                            if k != 'seconds': api.base.core.exact(direct[k], state[k])
                        pilot_states.append(state)
                    else:
                        receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                        once(receipt, dict(context=context, seed=seed, arm=arm, step=state['step'],
                            checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),
                            seconds=state['seconds'],parameters=state['parameters'],episode_hash=state['episode_hash'],
                            first_trace=state['trace'][0],final_trace=state['trace'][-1],
                            meta_forward_rows=state['meta_forward_rows'],registration_sha256=sha(REG),
                            result_source='fresh_run_neural_policy',second_order_gradient=True,
                            new_forecaster=False,inner_held_used=False,test_time_adaptation=False))
                        receipts.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)))
                        beat(state='fit_complete',fits=len(receipts),total=36)
                if len(set(episode_hashes)) != 1: raise ValueError('Episode streams must match across arms')
                del data, parent
        if pilot:
            once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-began,actual_updates=80,
                estimated_fit_seconds=sum(s['seconds'] for s in pilot_states)*100*18,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                exact_resume_both_arms=True,engineering_pass=True,registration_sha256=sha(REG)))
        else:
            once(finish,dict(fits=receipts,count=36,optimizer_steps=72000,pid=os.getpid(),
                seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                registration_sha256=sha(REG),inner_held_scored=False,independent_roles_read=False))
        beat(state='pilot_complete' if pilot else 'training_complete')


def models():
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); out = {}
    if freeze['count'] != 36 or freeze['registration_sha256'] != sha(REG) or freeze['inner_held_scored']:
        raise ValueError('All36 finals frozen before readout')
    for ref in freeze['fits']:
        path = ROOT/ref['path']; row = json.loads(path.read_text())
        if sha(path) != ref['sha256'] or sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen fit changed')
        key = row['context'],row['seed'],row['arm']
        if key in out or row['step'] != 2000: raise ValueError('Duplicate or incomplete fit')
        out[key] = row
    if len(out) != 36: raise ValueError('Complete roster required')
    return out


def evaluate(check=False, resume=False):
    reg, cfg, ancestor = admission(); learned = models(); parents = previous.models()
    _, origins = previous.previous.ORIGINS.models(reg); finish = PUBLIC/'complete.json'
    if finish.exists() and not check: raise ValueError('Completed readout immutable')
    prior = json.loads((previous.PUBLIC/'complete.json').read_text())
    prior_refs = {Path(r['path']).name:r for r in prior['groups']}
    began, beat, guard = runtime(cfg); rows, refs = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, held = previous.previous.LOADER.packets(items, ancestor, beat)
            for seed in reg['seeds']:
                origin = api.base.core.read_checkpoint(ROOT/origins[context,seed,'self']['checkpoint']['path'])
                states = {a:api.base.core.read_checkpoint(ROOT/learned[context,seed,a]['checkpoint']['path']) for a in api.ARMS}
                states['exact_gradient'] = api.base.core.read_checkpoint(ROOT/parents[context,seed,'exact_gradient']['checkpoint']['path'])
                for scope, packets in (('fit',fit),('inner_held',held)):
                    predictions = previous.previous.origin_predictions(origin, packets)
                    for source, args in sorted(packets.items()):
                        path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{scope}.json'
                        if path.exists() and not (check or resume): raise ValueError('Explicit resume required')
                        eligible = api.base.shared.eligible(args)
                        ranks = dict(floor=api.np.zeros(len(args[0])),origin=predictions[source][:,0]-predictions[source][:,1])
                        masks = dict(floor=api.np.zeros(len(args[0]),bool),origin=api.base.actions(predictions[source],eligible))
                        for arm, state in states.items():
                            ranks[arm],masks[arm] = api.base.predict(state,source,args[0],args[1],predictions[source],eligible)
                        result = readout.evaluate(args,masks,ranks)
                        oldpath = previous.PUBLIC/'groups'/path.name
                        if sha(oldpath) != prior_refs[path.name]['sha256']: raise ValueError('Prior group changed')
                        old = json.loads(oldpath.read_text())
                        for arm in ('origin','exact_gradient'):
                            if (api.base.fingerprint(masks[arm]) != old['action_sha256'][arm]
                                    or api.base.fingerprint(ranks[arm]) != old['score_sha256'][arm]
                                    or result['policies'][arm] != old['result']['policies'][arm]):
                                raise ValueError('Frozen reference mismatch')
                        row = dict(context=context,source=source,seed=seed,scope=scope,result=result,
                            registration_sha256=sha(REG),training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                            parent_group_sha256=sha(oldpath),score_sha256={k:api.base.fingerprint(v) for k,v in ranks.items()},
                            action_sha256={k:api.base.fingerprint(v) for k,v in masks.items()},
                            result_source='fresh_run_fixed_final_TRAIN_internal',independent_roles_read=False,
                            deployment_changed=False,whole_pipeline_OOF=False)
                        if check:
                            if row != json.loads(path.read_text()): raise ValueError('Readout replay differs')
                        else: once(path,row)
                        rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                beat(state='readout_progress',views=len(rows),total=144);guard()
        summary = readout.summarize(rows)
        if check:
            old = json.loads(finish.read_text())
            if old['groups'] != refs or old['summary_sha256'] != sha(PUBLIC/'summary.json') or summary != json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json',summary)
            once(finish,dict(groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),registration_sha256=sha(REG),
                training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),pid=os.getpid(),seconds=time.monotonic()-began,
                independent_roles_read=False))
        beat(state='verified' if check else 'readout_complete',views=144,budget=summary['primary_full_budget_supported'])


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action',choices=('register','pilot','train','evaluate','check'))
    p.add_argument('--resume',action='store_true');a = p.parse_args()
    if a.action == 'register': register()
    elif a.action in ('pilot','train'): train(a.action == 'pilot',a.resume)
    else: evaluate(a.action == 'check',a.resume)
