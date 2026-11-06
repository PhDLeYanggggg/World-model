"""Replicate donor information at fixed teacher/data, not a new held test."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_donor_augmentation as old
from src.world_model import m3w_seeded_donor_risk as method

np, torch = old.np, old.torch
read, stable, sha, once, checked, artifact = old.read, old.stable, old.sha, old.once, old.checked, old.artifact
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_donor_seeds_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_donor_seeds_v1'
CONFIG = ROOT/'configs/m3w_moving_donor_seeds_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
         ROOT/'src/world_model/m3w_seeded_donor_risk.py', ROOT/'tests/test_m3w_seeded_donor_risk.py')
METRICS = ('normalized_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost')


def register():
    cfg = read(CONFIG)
    _, parent, previous = old.register()
    frozen = read(old.PUBLIC/'analysis.json')
    if (cfg['risk_head_seeds'] != list(method.SEEDS) or cfg['fresh_seeds'] != [29, 43]
            or cfg['frozen_forecaster_seed'] != 17 or cfg['arms'] != list(method.original.ARMS)
            or cfg['steps_per_model'] != 4096 or cfg['fresh_models'] != 48
            or cfg['risk_budget'] != .02 or cfg['minimum_cost_reduction'] != .01
            or cfg['minimum_contexts_per_seed'] != 2 or cfg['bootstrap_resamples'] != 3000
            or cfg['bootstrap_seed'] != 62027 or any(cfg[k] for k in
                ('threshold_search', 'new_outer_C_H_readout', 'identity_certification',
                 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Keep fixed replication scope and original rules')
    if not read(old.PUBLIC/'verification.json')['all24_full_refits_exact']:
        raise ValueError('Verified frozen seed17 models required')
    value = dict(config=cfg, files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_files={str(p.relative_to(ROOT)): sha(p) for p in (*old.FILES, Path(method.original.parent.__file__))},
        parent_registration=artifact(old.REG), cached_seed17_result=artifact(old.PUBLIC/'analysis.json'),
        frozen_donors=artifact(old.PUBLIC/'donors.json'),
        roles='fixed_teacher17_risk_head_seed_replication_on_exposed_own_B',
        independent_confirmation=False, new_outer_C_H_readout=False, original_rules_unchanged=True)
    once(REG, stable(value))
    return cfg, parent, previous, frozen


def committed():
    for p in (*FILES, REG):
        content = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                                 capture_output=True, check=True).stdout
        if hashlib.sha256(content).hexdigest() != sha(p):
            raise ValueError('Commit registered experiment before real fitting')


def runtime(cfg, phase):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    start, last = time.monotonic(), [0.]
    def guard():
        now = time.monotonic()
        if now-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('Resume same experiment after12h')
        if now-last[0] > 1:
            last[0] = now
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
                raise OSError('Keep5GiB disk reserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['artifact_cap_bytes']:
                raise OSError('New private artifact cap exceeded')
    def beat(**values):
        guard()
        event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
        old.cross.atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    beat(state='ready', machine=platform.machine(), device='cpu', torch_threads=4, interop_threads=1, workers=0)
    return start, guard, beat


def identity(parent_identity, seed):
    return dict(registration_sha256=sha(REG), frozen_data_binding=parent_identity, risk_head_seed=seed)


def path_for(c, f, arm, seed, kind='fits'):
    return PRIVATE/kind/f'{c["job"]}_{f["source"]}_{arm}_seed{seed}.pt.gz'


def pilot(parent, previous, guard, beat):
    c, f, arm, data, binding, own = next(e for e in old.entries(parent, previous) if e[2]=='augmented')
    original = method.original.parent.storage.base.read_checkpoint(old.fit_path(c, f, arm))
    a = method.fit(data, f['source'], binding, PRIVATE/'pilot/equivalent17.pt.gz', seed=17,
                   guard=guard, heartbeat=beat)
    method.original.exact(a, original)
    b = identity(binding, 29)
    p, q = PRIVATE/'pilot/resumed29.pt.gz', PRIVATE/'pilot/direct29.pt.gz'
    method.fit(data, f['source'], b, p, seed=29, stop_at=128, guard=guard, heartbeat=beat)
    resumed = method.fit(data, f['source'], b, p, seed=29, stop_at=256, resume=True, guard=guard, heartbeat=beat)
    direct = method.fit(data, f['source'], b, q, seed=29, stop_at=256, guard=guard, heartbeat=beat)
    method.original.exact(resumed, direct)
    once(PUBLIC/'pilot.json', stable(dict(full4096update_engine_seed17_exact=True,
        seed29_128plus128_resume_exact=True, pilot_updates=4608, fit_rows=a['fit_rows'],
        seconds=a['seconds'], estimated48_primary_fit_seconds=a['seconds']*48,
        pilot_not_independent_evidence=True)))


def train(cfg, parent, previous, guard, beat, resume=False, verify=False):
    refs = []
    count = 0
    for c, f, arm, data, binding, own in old.entries(parent, previous):
        for seed in cfg['fresh_seeds']:
            path = path_for(c, f, arm, seed, 'verify' if verify else 'fits')
            state = method.fit(data, f['source'], identity(binding, seed), path,
                seed=seed, resume=resume, guard=guard,
                heartbeat=lambda **kw: beat(job=c['job'], source=f['source'], arm=arm, seed=seed, **kw))
            if not state['complete'] or state['step'] != 4096 or int(state['visits'].sum()) != 4096*512:
                raise ValueError('Complete matched optimizer and sample budget required')
            if f['source'] in state['preprocess']['train_sources']:
                raise ValueError('Held source in fitting bank')
            if verify:
                method.original.exact(method.original.parent.storage.base.read_checkpoint(path_for(c, f, arm, seed)), state)
            else:
                receipt = PUBLIC/'fits'/f'{path.stem}.json'
                once(receipt, stable(dict(job=c['job'], source=f['source'], arm=arm, seed=seed,
                    checkpoint=artifact(path), parameters=state['parameters'], steps=state['step'],
                    seconds=state['seconds'], trace=state['trace'], fit_rows=state['fit_rows'],
                    distinct_fitting_rows_drawn=int((state['visits']>0).sum()),
                    fitting_sources=state['preprocess']['train_sources'], result_source='fresh_run')))
                refs.append(artifact(receipt))
            count += 1
            beat(state='fit_verified' if verify else 'fit_complete', complete=count, total=48)
    if verify:
        once(PUBLIC/'verification.json', dict(all48_full_refits_exact=True, fresh_repeat_updates=196608))
    else:
        once(PUBLIC/'training_freeze.json', dict(fits=refs, fresh_models=48, fresh_updates=196608,
             cached_seed17_models=24, all_before_final_new_seed_readout=True))


def summarize(records, frozen):
    contexts, passing = [], {}
    for seed in method.SEEDS:
        count = 0
        for parent in frozen['contexts']:
            job = parent['job']
            subset = [r for r in records if r['job']==job and r['seed']==seed]
            means = {arm: {k: float(np.mean([r['metrics'][k] for r in subset if r['arm']==arm]))
                           for k in METRICS} for arm in method.original.ARMS}
            own, aug, forest = means['own_only'], means['augmented'], parent['source_macro']['shared64']
            cost_gain = 1-aug['total_joint_cost']/own['total_joint_cost'] if own['total_joint_cost'] else None
            mse_gain = 1-aug['normalized_MSE']/own['normalized_MSE'] if own['normalized_MSE'] else None
            passed = bool(cost_gain is not None and cost_gain>=.01 and mse_gain is not None and mse_gain>0
                          and aug['false_safe_cost']<=own['false_safe_cost'])
            count += passed
            contexts.append(dict(seed=seed, job=job, source_macro=means,
                cost_reduction_vs_own=cost_gain, MSE_reduction_vs_own=mse_gain,
                cost_reduction_vs_forest=1-aug['total_joint_cost']/forest['total_joint_cost'],
                false_safe_cost_nonworse_forest=bool(aug['false_safe_cost']<=forest['false_safe_cost']),
                passes_information_screen=passed, safety_certificate=False))
        passing[str(seed)] = count
    cost, own_cost, forest_cost = [], [], []
    for parent in frozen['contexts']:
        job = parent['job']
        sites = sorted({r['source'] for r in records if r['job']==job})
        a, b, f = [], [], []
        for seed in method.SEEDS:
            a.append([next(r for r in records if r['job']==job and r['seed']==seed and r['source']==s
                           and r['arm']=='augmented')['metrics']['total_joint_cost'] for s in sites])
            b.append([next(r for r in records if r['job']==job and r['seed']==seed and r['source']==s
                           and r['arm']=='own_only')['metrics']['total_joint_cost'] for s in sites])
            # Cache the per-source forest metric, not a context-wide constant.
            f.append([next(r for r in records if r['job']==job and r['source']==s)['forest_metrics']['total_joint_cost'] for s in sites])
        cost.append(np.asarray(a)); own_cost.append(np.asarray(b)); forest_cost.append(np.asarray(f))
    return dict(contexts=contexts, passing_contexts_by_seed=passing,
        replication_screen_pass=all(n>=2 for n in passing.values()),
        source_bootstrap_vs_own=method.bootstrap_costs(cost, own_cost),
        source_bootstrap_vs_forest=method.bootstrap_costs(cost, forest_cost),
        fresh_models=48, cached_seed17_models=24, forecaster_seeds=[17], risk_head_seeds=list(method.SEEDS),
        all72_predictions_evaluated=True, original_rules_unchanged=True,
        new_outer_C_H_readout=False, independent_confirmation=False, identity_certification=False,
        original24group_primary='not_estimable_not_passed', deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def evaluate(cfg, parent, previous, frozen, beat, replay=False):
    if not read(PUBLIC/'verification.json')['all48_full_refits_exact']:
        raise ValueError('Full refits before readout')
    records = []
    for c, f, arm, data, binding, own in old.entries(parent, previous):
        held = own['sites'] == f['source']
        for seed in method.SEEDS:
            path = old.fit_path(c, f, arm) if seed==17 else path_for(c, f, arm, seed)
            if seed==17:
                ref = read(old.PUBLIC/'fits'/f'{c["job"]}_{f["source"]}_{arm}.json')['checkpoint']
                checked(ref)
            else:
                checked(read(PUBLIC/'fits'/f'{path.stem}.json')['checkpoint'])
            state = method.original.parent.storage.base.read_checkpoint(path)
            method.original.parent.storage.base.exact(state['identity'], binding if seed==17 else identity(binding, seed))
            if state['step']!=4096 or not state['complete']:
                raise ValueError('Only fixed final checkpoints')
            prediction = method.original.parent.predict(state, own['x'][held])
            metrics = method.original.parent.base.metrics(prediction, own['target'][held],
                         own['weights'][held], own['target_scale'][held][0])
            if seed==17:
                cached = next(r for r in frozen['folds'] if r['job']==c['job'] and r['source']==f['source'] and r['arm']==arm)
                method.original.parent.storage.base.exact(metrics, cached['metrics'])
                method.original.parent.storage.base.exact(method.original.parent.digest(prediction), cached['prediction_hash'])
            out = PRIVATE/'predictions'/f'{c["job"]}_{f["source"]}_{arm}_seed{seed}.npy'
            if replay:
                np.testing.assert_array_equal(np.load(out, allow_pickle=False), prediction)
            else:
                out.parent.mkdir(exist_ok=True)
                if out.exists():
                    np.testing.assert_array_equal(np.load(out, allow_pickle=False), prediction)
                else:
                    with out.open('wb') as stream:
                        np.save(stream, prediction, allow_pickle=False)
            records.append(dict(seed=seed, job=c['job'], source=f['source'], arm=arm,
                rows=int(held.sum()), metrics=metrics, forest_metrics=f['metrics']['shared64'],
                prediction_hash=method.original.parent.digest(prediction),
                result_source='cached_verified' if seed==17 else 'fresh_run'))
    analysis = stable(dict(folds=records, **summarize(records, frozen)))
    if replay:
        method.original.parent.storage.base.exact(read(PUBLIC/'analysis.json'), analysis)
    else:
        once(PUBLIC/'analysis.json', analysis)
    beat(state='replay_exact' if replay else 'new_seed_final_readout_complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'pilot', 'train', 'verify', 'evaluate', 'replay'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, parent, previous, frozen = register()
        if args.phase=='register':
            print('Registered48new risk fits; fixed teacher17, no new outer readout.')
            return
        committed()
        start, guard, beat = runtime(cfg, args.phase)
        if args.phase=='pilot':
            pilot(parent, previous, guard, beat)
        elif args.phase in ('train', 'verify'):
            if not read(PUBLIC/'pilot.json')['full4096update_engine_seed17_exact']:
                raise ValueError('Real engine equivalence required')
            train(cfg, parent, previous, guard, beat, args.resume, args.phase=='verify')
        else:
            before = {str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
            if args.phase=='replay':
                def forbidden(*a, **kw):
                    raise RuntimeError('Fitting forbidden in replay')
                method.fit = forbidden
            evaluate(cfg, parent, previous, frozen, beat, args.phase=='replay')
            if args.phase=='replay':
                method.original.parent.storage.base.exact(before, {str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        once((PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json'),
             dict(pid=os.getpid(), seconds=time.monotonic()-start,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        beat(state='phase_complete')


if __name__ == '__main__':
    main()
