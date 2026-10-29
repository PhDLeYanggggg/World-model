"""Frozen TRAIN diagnosis and whole-record neural policy experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import time

from scripts import run_m3w_direct_policy as previous
from src.world_model import m3w_whole_record_policy as api
from src.evaluation import m3w_whole_record_readout as readout

ROOT, sha, once = previous.ROOT, previous.sha, previous.once
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_whole_record_policy_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_whole_record_policy_v1'
CONFIG = ROOT/'configs/m3w_whole_record_policy_v1.json'
DIAG_REG, REG = PUBLIC/'diagnostic_registration.json', PUBLIC/'registration.json'


def register(training=False):
    reg, old_cfg, ancestor = previous.admission(); cfg = json.loads(CONFIG.read_text())
    settings = {k:v for k,v in cfg['training'].items() if k != 'chunk_size'}
    if settings != old_cfg['training'] or cfg['risk_budget'] != .02 or cfg['head_seeds'] != reg['seeds']:
        raise ValueError('Keep original architecture, update budget, thresholds and settings')
    paths = [Path(__file__).resolve(), CONFIG, PUBLIC/'protocol.md', PUBLIC/'diagnostic_registration_draft.json',
        ROOT/'src/world_model/m3w_whole_record_policy.py', ROOT/'src/evaluation/m3w_whole_record_readout.py',
        ROOT/'tests/test_m3w_whole_record_policy.py', ROOT/'tests/test_m3w_whole_record_readout.py',
        previous.REG, previous.PUBLIC/'training_freeze.json', previous.PUBLIC/'complete.json',
        previous.PUBLIC/'summary.json']
    if training: paths += [DIAG_REG, PUBLIC/'diagnostic_complete.json', PUBLIC/'diagnostic_summary.json']
    target = REG if training else DIAG_REG
    once(target, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], primary='exact_gradient',
        stage='training' if training else 'frozen_TRAIN_diagnosis', fits=36 if training else 0,
        diagnostic_states=18, steps_per_fit=2000, expected_views_per_scope=72,
        inference_threshold=0., inner_held_reused_exploratory=True,
        independent_roles_read=False, whole_pipeline_OOF=False, deployment_changed=False))
    return reg, cfg, ancestor


def admission(training=False):
    out = register(training); target = REG if training else DIAG_REG
    for rel in [*json.loads(target.read_text())['bindings'], str(target.relative_to(ROOT))]:
        previous.previous.pooled.previous.previous.committed(ROOT/rel)
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
        p = PRIVATE/'heartbeat.tmp'; p.write_text(json.dumps(event)+'\n'); os.replace(p, PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps({k:v for k,v in event.items() if k not in ('dual','exact_record_risk')}), flush=True)
        if time.monotonic()-began > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume identical scope')
    guard(); beat(state='admitted')
    return began, beat, guard


def prepare(context, seed, packets, origins, parents):
    origin_ref = origins[context, seed, 'self']['checkpoint']
    origin = api.base.core.read_checkpoint(ROOT/origin_ref['path'])
    predictions = previous.origin_predictions(origin, packets)
    previous.verify_original(context, seed, packets, predictions)
    parent_ref = parents[context, seed, 'hard_record']['checkpoint']
    parent = api.base.core.read_checkpoint(ROOT/parent_ref['path'])
    data = api.base.prepare(packets, predictions)
    for key in ('input_hashes','source_preprocess','record_keys','record_scales'):
        api.base.core.exact(data[key], parent[key])
    api.base.core.exact(data['sizes'], parent['record_sizes'])
    return data, parent, origin_ref, parent_ref


def diagnose(resume=False):
    reg, cfg, ancestor = admission(); _, origins = previous.ORIGINS.models(reg); parents = previous.models()
    finish = PUBLIC/'diagnostic_complete.json'
    if finish.exists(): raise ValueError('Completed diagnosis immutable')
    start, beat, guard = runtime(cfg); rows, refs = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            packets, held = previous.LOADER.packets(items, ancestor, beat); del held
            for seed in reg['seeds']:
                data, parent, _, ref = prepare(context, seed, packets, origins, parents)
                path = PUBLIC/'diagnostics'/f'{context}_seed{seed}.json'
                if path.exists():
                    if not resume: raise ValueError('Explicit resume required')
                    row = json.loads(path.read_text())
                    if row['checkpoint_sha256'] != ref['sha256'] or row['registration_sha256'] != sha(DIAG_REG):
                        raise ValueError('Diagnostic provenance changed')
                else:
                    row = dict(context=context, seed=seed, checkpoint_sha256=ref['sha256'],
                        registration_sha256=sha(DIAG_REG), result=api.diagnose(data, parent),
                        result_source='fresh_run_frozen_TRAIN_diagnostic', independent_roles_read=False)
                    once(path, row)
                if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Frozen model changed')
                rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='diagnosed', views=len(rows), total=18); guard()
                del data, parent
        positive = [(rec, k) for r in rows for rec in r['result']['records']
                    for k in (0,1) if rec['exact_mean'][k] > 0]
        cosine = [v for r in rows for v in r['result']['sample_full_gradient_cosines'] if v is not None]
        summary = dict(states=18, positive_record_channels=len(positive),
            positive_channels_batch_nonpositive_at_least_half=sum(rec['nonpositive_sample_fraction'][k] >= .5 for rec,k in positive),
            median_nonpositive_draw_fraction=statistics.median(rec['nonpositive_sample_fraction'][k] for rec,k in positive) if positive else None,
            median_miss_positive_probability=statistics.median(rec['miss_positive_probability'][k] for rec,k in positive) if positive else None,
            sample_full_gradient_cosine_median=statistics.median(cosine) if cosine else None,
            negative_gradient_cosines=sum(v < 0 for v in cosine), defined_gradient_cosines=len(cosine),
            no_optimizer_step=True, independent_roles_read=False)
        once(PUBLIC/'diagnostic_summary.json', summary)
        once(finish, dict(groups=refs, summary_sha256=sha(PUBLIC/'diagnostic_summary.json'),
            registration_sha256=sha(DIAG_REG), pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, independent_roles_read=False))
        beat(state='diagnostic_complete', **summary)


def path_for(context, seed, arm, pilot=False):
    return PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}.pt.gz'


def train(pilot=False, resume=False):
    reg, cfg, ancestor = admission(True); _, origins = previous.ORIGINS.models(reg); parents = previous.models()
    finish = PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists(): raise ValueError('Completed phase immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real pilot required')
    start, beat, guard = runtime(cfg); receipts, pilot_states = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items()):
            packets, held = previous.LOADER.packets(items, ancestor, beat); del held
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                data, parent, origin_ref, parent_ref = prepare(context, seed, packets, origins, parents)
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
                    if arm == 'exact_dual' and not pilot:
                        for key in ('draw_hash','row_draws','sampler_rng'):
                            api.base.core.exact(state[key], parent[key])
                    if pilot:
                        direct_path = PRIVATE/'pilot'/f'{arm}_direct.pt.gz'
                        direct = api.fit(**{**kw,'path':direct_path}, stop_at=20, resume=resume and direct_path.exists())
                        for k in direct:
                            if k != 'seconds': api.base.core.exact(direct[k], state[k])
                        pilot_states.append(state)
                    else:
                        receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                        once(receipt, dict(context=context, seed=seed, arm=arm, step=state['step'],
                            checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                            seconds=state['seconds'], parameters=state['parameters'],
                            first_trace=state['trace'][0], final_trace=state['trace'][-1],
                            row_draws=state['row_draws'], whole_record_forward_rows=state['whole_record_forward_rows'],
                            full_gradient_rows=state['full_gradient_rows'], registration_sha256=sha(REG),
                            result_source='fresh_run_neural_policy', new_forecaster=False, inner_held_used=False))
                        receipts.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                        beat(state='fit_complete', fits=len(receipts), total=36)
                del data, parent
        if pilot:
            once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-start, actual_updates=80,
                estimated_fit_seconds=sum(s['seconds'] for s in pilot_states)*100*18,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                exact_resume_both_arms=True, engineering_pass=True, registration_sha256=sha(REG)))
        else:
            once(finish, dict(fits=receipts, count=36, optimizer_steps=72000,
                pid=os.getpid(), seconds=time.monotonic()-start,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                registration_sha256=sha(REG), inner_held_scored=False, independent_roles_read=False))
        beat(state='pilot_complete' if pilot else 'training_complete')


def models():
    freeze = json.loads((PUBLIC/'training_freeze.json').read_text()); out = {}
    if freeze['count'] != 36 or freeze['registration_sha256'] != sha(REG) or freeze['inner_held_scored']:
        raise ValueError('All36 finals frozen before readout')
    for ref in freeze['fits']:
        path = ROOT/ref['path']; d = json.loads(path.read_text())
        if sha(path) != ref['sha256'] or sha(ROOT/d['checkpoint']['path']) != d['checkpoint']['sha256']:
            raise ValueError('Frozen fit changed')
        key = d['context'], d['seed'], d['arm']
        if key in out or d['step'] != 2000: raise ValueError('Duplicate or incomplete fit')
        out[key] = d
    if len(out) != 36: raise ValueError('Complete fit roster required')
    return out


def evaluate(check=False, resume=False):
    reg, cfg, ancestor = admission(True); learned = models(); parents = previous.models()
    _, origins = previous.ORIGINS.models(reg); finish = PUBLIC/'complete.json'
    if finish.exists() and not check: raise ValueError('Completed readout immutable')
    prior = json.loads((previous.PUBLIC/'complete.json').read_text())
    prior_refs = {Path(r['path']).name:r for r in prior['groups']}
    start, beat, guard = runtime(cfg); rows, refs = [], []
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, held = previous.LOADER.packets(items, ancestor, beat)
            for seed in reg['seeds']:
                origin = api.base.core.read_checkpoint(ROOT/origins[context, seed, 'self']['checkpoint']['path'])
                states = {a:api.base.core.read_checkpoint(ROOT/learned[context, seed, a]['checkpoint']['path']) for a in api.ARMS}
                states['stochastic'] = api.base.core.read_checkpoint(ROOT/parents[context, seed, 'hard_record']['checkpoint']['path'])
                for scope, packets in (('fit',fit), ('inner_held',held)):
                    predictions = previous.origin_predictions(origin, packets)
                    for source, args in sorted(packets.items()):
                        path = PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{scope}.json'
                        if path.exists() and not (check or resume): raise ValueError('Explicit resume required')
                        eligible = api.base.shared.eligible(args)
                        ranks = dict(floor=api.np.zeros(len(args[0])), origin=predictions[source][:,0]-predictions[source][:,1])
                        masks = dict(floor=api.np.zeros(len(args[0]),bool), origin=api.base.actions(predictions[source],eligible))
                        for arm, state in states.items():
                            ranks[arm], masks[arm] = api.base.predict(state, source, args[0], args[1], predictions[source], eligible)
                        result = readout.evaluate(args, masks, ranks)
                        oldpath = previous.PUBLIC/'groups'/path.name
                        if sha(oldpath) != prior_refs[path.name]['sha256']: raise ValueError('Prior group changed')
                        old = json.loads(oldpath.read_text())
                        for new, former in (('origin','origin'), ('stochastic','hard_record')):
                            if (api.base.fingerprint(masks[new]) != old['action_sha256'][former]
                                    or api.base.fingerprint(ranks[new]) != old['score_sha256'][former]
                                    or result['policies'][new] != old['result']['policies'][former]):
                                raise ValueError('Frozen reference does not replay')
                        row = dict(context=context, source=source, seed=seed, scope=scope, result=result,
                            registration_sha256=sha(REG), training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                            parent_group_sha256=sha(oldpath), score_sha256={k:api.base.fingerprint(v) for k,v in ranks.items()},
                            action_sha256={k:api.base.fingerprint(v) for k,v in masks.items()},
                            result_source='fresh_run_fixed_final_TRAIN_internal', independent_roles_read=False,
                            deployment_changed=False, whole_pipeline_OOF=False)
                        if check:
                            if row != json.loads(path.read_text()): raise ValueError('Readout replay differs')
                        else: once(path,row)
                        rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                beat(state='readout_progress', views=len(rows), total=144); guard()
        summary = readout.summarize(rows)
        if check:
            old = json.loads(finish.read_text())
            if old['groups'] != refs or old['summary_sha256'] != sha(PUBLIC/'summary.json') or summary != json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json',summary)
            once(finish,dict(groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),registration_sha256=sha(REG),
                training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),pid=os.getpid(),seconds=time.monotonic()-start,
                independent_roles_read=False))
        beat(state='verified' if check else 'readout_complete', views=144, budget=summary['primary_full_budget_supported'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('register-diagnostic','diagnose','register','pilot','train','evaluate','check'))
    p.add_argument('--resume',action='store_true'); a = p.parse_args()
    if a.action.startswith('register'): register(a.action == 'register')
    elif a.action == 'diagnose': diagnose(a.resume)
    elif a.action in ('pilot','train'): train(a.action == 'pilot',a.resume)
    else: evaluate(a.action == 'check',a.resume)
