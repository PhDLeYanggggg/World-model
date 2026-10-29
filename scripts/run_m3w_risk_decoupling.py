"""Registered fitting-only objective intervention with protected utility outputs."""
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

from scripts import run_m3w_query_context_risk as parent
from src.world_model import m3w_risk_decoupling as train
from src.evaluation import m3w_risk_decoupling_readout as readout

ROOT, sha, once = parent.ROOT, parent.sha, parent.once
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_risk_decoupling_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_risk_decoupling_v1'
REG = PUBLIC/'registration.json'
UPDATES = 2000


def register():
    reg, cfg, ancestor = parent.admission()
    files = [Path(__file__), ROOT/'src/world_model/m3w_risk_decoupling.py',
             ROOT/'src/evaluation/m3w_risk_decoupling_readout.py',
             ROOT/'tests/test_m3w_risk_decoupling.py', PUBLIC/'protocol.md', PUBLIC/'prior_controls.md']
    once(REG, dict(bindings={str(p.resolve().relative_to(ROOT)): sha(p) for p in files},
        parent_registration_sha256=sha(parent.REG), parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        conditioning_summary_sha256=sha(PUBLIC.parent/'european_query_conditioning_v2/summary.json'),
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], objective_arms=list(train.ARMS),
        fresh_fits=36, updates_per_fit=UPDATES, primary_cohort='frozen_original_self_selection',
        held_inference=False, independent_roles_read=False, threshold_search=False,
        deployment_changed=False, new_forecaster=False))
    return reg, cfg, ancestor


def committed(path):
    p = subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True)
    if p.returncode or p.stdout != path.read_bytes(): raise ValueError('Commit frozen artifact before execution')


def admission():
    result = register(); doc = json.loads(REG.read_text())
    for rel in doc['bindings']: committed(ROOT/rel)
    committed(REG)
    return result


def runtime():
    train.torch.set_num_threads(4); train.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); began = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume saved atomic checkpoint')))
    def beat(**kw):
        value = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), **kw)
        temp = PRIVATE/'heartbeat.tmp'; temp.write_text(json.dumps(value)+'\n'); os.replace(temp, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
        if time.monotonic()-began > 43200: raise TimeoutError('12h boundary; explicit resume without scope reduction')
    return began, beat


def storage():
    if shutil.disk_usage(ROOT).free < 10*2**30 or sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > 256*2**20:
        raise OSError('Preserve10GiB reserve and256MiB private checkpoint cap')


def fit(phase, resume):
    reg, _, ancestor = admission(); _, origins = parent.models(reg)
    finish = PUBLIC/('pilot.json' if phase == 'pilot' else 'training_freeze.json')
    if finish.exists(): raise ValueError('Completed phase immutable')
    if phase == 'train' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real pilot required')
    began, beat = runtime(); refs = []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); storage()
        contexts = sorted(reg['contexts'])[:1] if phase == 'pilot' else sorted(reg['contexts'])
        for context in contexts:
            packets, held = parent.parent.run.parent.prior.packets(reg['contexts'][context], ancestor, beat); del held
            for seed in reg['seeds'][:1] if phase == 'pilot' else reg['seeds']:
                ref = origins[context, seed, 'self']['checkpoint']
                origin = train.api.core.read_checkpoint(ROOT/ref['path']); states = []
                for arm in train.ARMS:
                    key = f'{context}_head{seed}_{arm}'
                    path = PRIVATE/('pilot' if phase == 'pilot' else 'heads')/(key+'.pt.gz')
                    kw = dict(origin=origin, origin_sha256=ref['sha256'], objective_arm=arm, updates=UPDATES,
                              path=path, registration_sha256=sha(REG), heartbeat=beat, guard=storage)
                    if phase == 'pilot' and arm == 'estimation' and not path.exists():
                        train.fit(packets, stop_at=50, **kw)
                    state = train.fit(packets, stop_at=100 if phase == 'pilot' else None,
                        resume=path.exists() and (resume or phase == 'pilot'), **kw)
                    if not any(not train.torch.equal(v, origin['model'][k]) for k, v in state['model'].items()
                               if k.startswith('risk_encoder.')):
                        raise ValueError('Risk encoder did not update')
                    if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Original checkpoint changed')
                    states.append(state)
                    if phase == 'train':
                        out = PUBLIC/'fits'/(key+'.json')
                        once(out, dict(context=context, seed=seed, objective_arm=arm, continuation_updates=state['continuation_step'],
                            origin_checkpoint_sha256=ref['sha256'], registration_sha256=sha(REG),
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                            seconds=state['continuation_seconds'], first_trace=state['continuation_trace'][0],
                            final_trace=state['continuation_trace'][-1], trainable_parameters=state['trainable_parameters'],
                            risk_encoder_updated=True, result_source='fresh_run_real_Torch_continuation',
                            held_inference=False, independent_roles_read=False, new_forecaster=False))
                        refs.append(dict(path=str(out.relative_to(ROOT)), sha256=sha(out)))
                        beat(state='fit_complete', fits=len(refs), total=36)
                train.matched(*states)
                if phase == 'pilot':
                    direct = train.fit(packets, **{**kw, 'path': PRIVATE/'pilot/direct.pt.gz'}, stop_at=100, resume=resume)
                    for k in direct:
                        if k != 'continuation_seconds': train.api.core.exact(direct[k], states[-1][k])
                    for source, args in packets.items():
                        old = train.api.predict(origin, source, args[0], args[1], args[5], args[6])
                        for s in states:
                            p = train.api.predict(s, source, args[0], args[1], args[5], args[6])
                            train.np.testing.assert_array_equal(p[:, [0, 2, 3]], old[:, [0, 2, 3]])
                    once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-began, actual_updates=300,
                        architecture=platform.machine(), torch=train.torch.__version__, cpu_threads=4, workers=0,
                        exact_resume=True, protected_outputs_exact=True, matched_draws=True, engineering_pass=True,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        estimated_full_training_seconds=sum(s['continuation_seconds'] for s in states)*20*18*1.5,
                        registration_sha256=sha(REG), scientific_lift_established=False))
        if phase == 'train':
            if len(refs) != 36: raise ValueError('Complete36 continuations required')
            once(finish, dict(fits=refs, fresh_fits=36, optimizer_steps=72000, registration_sha256=sha(REG),
                pid=os.getpid(), seconds=time.monotonic()-began,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                held_inference=False, independent_roles_read=False, deployment_changed=False))
        beat(state=phase+'_complete')


def models(reg):
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); docs = {}
    if freeze['registration_sha256'] != sha(REG): raise ValueError('Training binding changed')
    for ref in freeze['fits']:
        path = ROOT/ref['path']; d = json.loads(path.read_text()); ck = ROOT/d['checkpoint']['path']
        if (path.is_symlink() or not path.resolve().is_relative_to((PUBLIC/'fits').resolve())
                or sha(path) != ref['sha256'] or ck.is_symlink()
                or not ck.resolve().is_relative_to((PRIVATE/'heads').resolve()) or sha(ck) != d['checkpoint']['sha256']
                or d['continuation_updates'] != UPDATES or d['registration_sha256'] != sha(REG)):
            raise ValueError('Changed frozen model')
        key = d['context'], d['seed'], d['objective_arm']
        if key in docs: raise ValueError('Duplicate fitted model')
        docs[key] = d
    if set(docs) != {(c, s, a) for c in reg['contexts'] for s in reg['seeds'] for a in train.ARMS}:
        raise ValueError('Incomplete frozen training roster')
    return docs


def evaluate(check, resume):
    reg, _, ancestor = admission(); committed(PUBLIC/'training_freeze.json')
    docs = models(reg); _, origins = parent.models(reg)
    if not check and (PUBLIC/'complete.json').exists(): raise ValueError('Completed fit readout immutable')
    began, beat = runtime(); rows, refs = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            packets, held = parent.parent.run.parent.prior.packets(items, ancestor, beat); del held
            for seed in reg['seeds']:
                states = {a: train.api.core.read_checkpoint(ROOT/docs[context, seed, a]['checkpoint']['path']) for a in train.ARMS}
                train.matched(states['coupled'], states['estimation'])
                states['origin'] = train.api.core.read_checkpoint(ROOT/origins[context, seed, 'self']['checkpoint']['path'])
                for source, args in sorted(packets.items()):
                    predictions = {a: train.api.predict(s, source, args[0], args[1], args[5], args[6]) for a, s in states.items()}
                    # Original prediction and fixed selection must match the preceding audit.
                    previous = json.loads((PUBLIC.parent/'european_query_conditioning_v2/groups'/
                        f'{context}_{source}_seed{seed}.json').read_text())
                    if train.api.fingerprint(predictions['origin']) != previous['prediction_hashes']['self']:
                        raise ValueError('Original frozen predictions changed')
                    result = readout.evaluate(args, predictions)
                    if result['common_cohort_hash'] != previous['common_mask_sha256']:
                        raise ValueError('Fixed fitting cohort changed')
                    row = dict(context=context, source=source, seed=seed, scope='fit', result=result,
                        prediction_hashes={a: train.api.fingerprint(p) for a, p in predictions.items()},
                        registration_sha256=sha(REG), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        result_source='fresh_run_frozen_inner_fit_readout', held_inference=False, independent_roles_read=False)
                    out = PUBLIC/'readout'/f'{context}_{source}_seed{seed}.json'
                    if check:
                        if json.loads(out.read_text()) != row: raise ValueError('Complete fit inference replay differs')
                    else:
                        if out.exists() and not resume: raise ValueError('Explicit readout resume required')
                        once(out, row)
                    rows.append(row); refs.append(dict(path=str(out.relative_to(ROOT)), sha256=sha(out)))
                    beat(state='verified_view' if check else 'fit_view', views=len(rows), total=72)
        summary = readout.summarize(rows, parent.expected(reg))
        if check:
            finish = json.loads((PUBLIC/'complete.json').read_text())
            if finish['readouts'] != refs or finish['summary_sha256'] != sha(PUBLIC/'summary.json') or summary != json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Frozen aggregate replay differs')
        else:
            once(PUBLIC/'summary.json', summary)
            once(PUBLIC/'complete.json', dict(readouts=refs, summary_sha256=sha(PUBLIC/'summary.json'),
                registration_sha256=sha(REG), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                seconds=time.monotonic()-began, pid=os.getpid(), held_inference=False,
                independent_roles_read=False, deployment_changed=False))
        beat(state='verified' if check else 'fit_readout_complete', advance=summary['fit_eligible_for_later_held_design'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('register', 'pilot', 'train', 'fit-eval', 'check'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    if args.action == 'register': register()
    elif args.action in ('pilot', 'train'): fit(args.action, args.resume)
    else: evaluate(args.action == 'check', args.resume)
