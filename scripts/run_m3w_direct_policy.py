"""Matched direct-policy training and fixed TRAIN-internal recording readout."""
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

from src.world_model import m3w_direct_policy as api
from src.evaluation import m3w_direct_policy_readout as readout
from scripts import run_m3w_easy_constrained as previous

ROOT, sha, once = previous.ROOT, previous.sha, previous.once
np = api.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_direct_policy_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_direct_policy_v1'
CONFIG = ROOT/'configs/m3w_direct_policy_v1.json'
REG = PUBLIC/'registration.json'
ORIGINS = previous.pooled.previous.previous.parent
LOADER = previous.pooled.LOADER


def register():
    reg, _, ancestor = previous.admission()
    cfg = json.loads(CONFIG.read_text())
    if cfg['head_seeds'] != reg['seeds'] or cfg['arms'] != list(api.ARMS) or cfg['risk_budget'] != .02:
        raise ValueError('Fixed seeds, arms and budget required')
    paths = [Path(__file__).resolve(), CONFIG, PUBLIC/'protocol.md',
        ROOT/'src/world_model/m3w_direct_policy.py', ROOT/'src/evaluation/m3w_direct_policy_readout.py',
        ROOT/'tests/test_m3w_direct_policy.py', ROOT/'tests/test_m3w_direct_policy_readout.py',
        previous.PUBLIC/'complete.json', previous.PUBLIC/'summary.json', previous.REG,
        ROOT/'src/evaluation/m3w_easy_constrained_policy.py',
        ROOT/'src/world_model/m3w_shared_train_risk.py',
        ROOT/'src/world_model/m3w_unknown_outcome_bounds.py',
        ROOT/'src/world_model/m3w_scaled_risk_readout.py',
        ROOT/'src/world_model/m3w_temporal_auxiliary_readout.py',
        ROOT/'src/world_model/m3w_component_calibration.py']
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], fits=54, steps_per_fit=2000,
        expected_views_per_scope=72, primary=cfg['primary'], inference_threshold=0.,
        inner_holdout_reused_exploratory=True, whole_pipeline_OOF=False,
        outer_development_read=False, independent_roles_read=False, deployment_changed=False))
    return reg, cfg, ancestor


def admission():
    out = register()
    for rel in [*json.loads(REG.read_text())['bindings'], str(REG.relative_to(ROOT))]:
        previous.pooled.previous.previous.committed(ROOT/rel)
    return out


def runtime(cfg):
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); began = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve registered disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Private model cap exceeded')
    def beat(**kw):
        event = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        path = PRIVATE/'heartbeat.tmp'; path.write_text(json.dumps(event)+'\n')
        os.replace(path, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps({k: v for k, v in event.items() if k not in ('dual', 'raw_record_risk', 'scaled_risk')}), flush=True)
        if time.monotonic()-began > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume same scope')
    guard(); beat(state='admitted')
    return began, beat, guard


def origin_predictions(state, packets):
    return {s: api.frozen.predict(state, s, a[0], a[1], a[5], a[6]) for s, a in packets.items()}


def verify_original(context, seed, packets, predictions):
    done = json.loads((previous.PUBLIC/'complete.json').read_text())
    refs = {Path(r['path']).name: r for r in done['groups']}
    for source, args in packets.items():
        p = previous.PUBLIC/'groups'/f'{context}_{source}_seed{seed}.json'
        if sha(p) != refs[p.name]['sha256']: raise ValueError('Prior policy receipt changed')
        old = json.loads(p.read_text()); action = api.actions(predictions[source], api.shared.eligible(args))
        if api.fingerprint(action) != old['action_sha256']['origin']:
            raise ValueError('Frozen original policy does not replay')
        if api.fingerprint(predictions[source][action]) != old['prediction_sha256']['origin']:
            raise ValueError('Frozen original moments do not replay')


def checkpoint_path(context, seed, arm, pilot=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}.pt.gz'


def train(phase, resume):
    reg, cfg, ancestor = admission(); _, origins = ORIGINS.models(reg)
    pilot = phase == 'pilot'; finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists(): raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real training/resume pilot required')
    start, beat, guard = runtime(cfg); receipts = []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, refs in sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items()):
            packets, held = LOADER.packets(refs, ancestor, beat); del held
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                ref = origins[context, seed, 'self']['checkpoint']
                origin = api.core.read_checkpoint(ROOT/ref['path'])
                predictions = origin_predictions(origin, packets)
                verify_original(context, seed, packets, predictions)
                data = api.prepare(packets, predictions); states = []
                identity = dict(context=context, seed=seed, registration_sha256=sha(REG), origin_sha256=ref['sha256'])
                for arm in api.ARMS:
                    path = checkpoint_path(context, seed, arm, pilot)
                    kw = dict(data=data, arm=arm, settings=cfg['training'], seed=seed, identity=identity,
                              path=path, heartbeat=beat, guard=guard)
                    if pilot and arm == 'hard_record' and not path.exists():
                        api.fit(**kw, stop_at=50)
                    state = api.fit(**kw, stop_at=100 if pilot else None,
                                    resume=path.exists() and (resume or pilot))
                    if not any(not api.torch.equal(v, state['initial_model'][k]) for k, v in state['model'].items()):
                        raise ValueError('No actual neural parameter updates')
                    states.append(state)
                    if not pilot:
                        p = PUBLIC/'fits'/(path.stem.replace('.pt', '')+'.json')
                        once(p, dict(context=context, seed=seed, arm=arm, step=state['step'],
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                            registration_sha256=sha(REG), seconds=state['seconds'], parameters=state['parameters'],
                            first_trace=state['trace'][0], final_trace=state['trace'][-1],
                            source_preprocess_fit_only=True, matched_initialization_and_draws=True,
                            result_source='fresh_run_neural_intervention_head', new_forecaster=False,
                            inner_held_used=False, independent_roles_read=False))
                        receipts.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
                        beat(state='fit_complete', fits=len(receipts), total=54)
                api.assert_matched(states)
                if pilot:
                    direct = api.fit(**{**kw, 'path': PRIVATE/'pilot/direct.pt.gz'}, stop_at=100,
                                     resume=resume and (PRIVATE/'pilot/direct.pt.gz').exists())
                    for key in direct:
                        if key != 'seconds': api.core.exact(direct[key], states[-1][key])
                    elapsed = time.monotonic()-start
                    once(finish, dict(pid=os.getpid(), seconds=elapsed, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        actual_updates=400, exact_resume=True, matched_draws=True, original_control_exact=True,
                        estimated_fit_seconds=sum(s['seconds'] for s in states)*20*18,
                        engineering_pass=True, registration_sha256=sha(REG), held_scored=False))
                    beat(state='pilot_complete'); return
                if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Frozen forecaster-risk artifact changed')
                del data, states, origin, predictions
        once(finish, dict(fits=receipts, count=54, optimizer_steps=108000, pid=os.getpid(),
            seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            registration_sha256=sha(REG), inner_held_scored=False, independent_roles_read=False))
        beat(state='training_complete', fits=54)


def models():
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); models = {}
    if freeze['registration_sha256'] != sha(REG) or freeze['count'] != 54 or freeze['inner_held_scored']:
        raise ValueError('All registered finals frozen before readout')
    for ref in freeze['fits']:
        path = ROOT/ref['path']; doc = json.loads(path.read_text())
        if sha(path) != ref['sha256'] or sha(ROOT/doc['checkpoint']['path']) != doc['checkpoint']['sha256']:
            raise ValueError('Checkpoint or receipt changed')
        key = doc['context'], doc['seed'], doc['arm']
        if key in models or doc['step'] != 2000: raise ValueError('Invalid frozen fit roster')
        models[key] = doc
    if len(models) != 54: raise ValueError('Complete fit roster required')
    return models


def evaluate(check=False, resume=False):
    reg, cfg, ancestor = admission(); learned = models(); _, origins = ORIGINS.models(reg)
    finish = PUBLIC/'complete.json'
    if finish.exists() and not check: raise ValueError('Completed fixed-final readout immutable')
    start, beat, guard = runtime(cfg); rows, receipts = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, refs in sorted(reg['contexts'].items()):
            fit, held = LOADER.packets(refs, ancestor, beat)
            for seed in reg['seeds']:
                origin = api.core.read_checkpoint(ROOT/origins[context, seed, 'self']['checkpoint']['path'])
                states = {a: api.core.read_checkpoint(ROOT/learned[context, seed, a]['checkpoint']['path']) for a in api.ARMS}
                api.assert_matched(list(states.values()))
                for scope, packets in (('fit', fit), ('inner_held', held)):
                    pp = origin_predictions(origin, packets)
                    if scope == 'fit': verify_original(context, seed, packets, pp)
                    for source, args in sorted(packets.items()):
                        path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{scope}.json'
                        if path.exists() and not (check or resume): raise ValueError('Explicit resume required')
                        eligible = api.shared.eligible(args)
                        ranks = dict(floor=np.zeros(len(args[0])), origin=pp[source][:, 0]-pp[source][:, 1])
                        masks = dict(floor=np.zeros(len(args[0]), bool), origin=api.actions(pp[source], eligible))
                        for arm in api.ARMS:
                            score, action = api.predict(states[arm], source, args[0], args[1], pp[source], eligible)
                            ranks[arm], masks[arm] = score, action
                        result = readout.evaluate(args, masks, ranks)
                        row = dict(context=context, source=source, seed=seed, scope=scope, result=result,
                            registration_sha256=sha(REG), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                            score_sha256={a: api.fingerprint(v) for a, v in ranks.items()},
                            action_sha256={a: api.fingerprint(v) for a, v in masks.items()},
                            result_source='fresh_run_fixed_final_TRAIN_internal', whole_pipeline_OOF=False,
                            independent_roles_read=False, deployment_changed=False)
                        if check:
                            if row != json.loads(path.read_text()): raise ValueError('Fixed-final replay differs')
                        else: once(path, row)
                        rows.append(row); receipts.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='readout_progress', views=len(rows), total=144)
            guard()
        summary = readout.summarize(rows)
        if check:
            old = json.loads(finish.read_text())
            if (old['groups'] != receipts or old['summary_sha256'] != sha(PUBLIC/'summary.json')
                    or summary != json.loads((PUBLIC/'summary.json').read_text())):
                raise ValueError('Full summary replay differs')
        else:
            once(PUBLIC/'summary.json', summary)
            once(finish, dict(groups=receipts, summary_sha256=sha(PUBLIC/'summary.json'), pid=os.getpid(),
                seconds=time.monotonic()-start, registration_sha256=sha(REG),
                training_freeze_sha256=sha(PUBLIC/'training_freeze.json'), independent_roles_read=False))
        beat(state='verified' if check else 'readout_complete', views=144,
             full_roster_budget_supported=summary['primary_full_roster_budget_supported'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('register', 'pilot', 'train', 'evaluate', 'check'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    if args.action == 'register': register()
    elif args.action in ('pilot', 'train'): train(args.action, args.resume)
    else: evaluate(check=args.action == 'check', resume=args.resume)
