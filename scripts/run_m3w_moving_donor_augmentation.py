"""Matched risk learning with source-held producer-A forecast donors."""
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
    raise RuntimeError('Use native arm64 .venv-pytorch before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_neural_transport as old
from scripts import run_m3w_moving_producer_crossfit as cross
from src.world_model import m3w_donor_augmentation as method
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.evaluation.m3w_amplitude_opportunity import FRACTIONS

np, torch = old.np, cross.torch
read, stable, sha, once, artifact, checked = cross.read, cross.stable, cross.sha, cross.once, cross.artifact, cross.checked
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_donor_augmentation_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_donor_augmentation_v1'
CONFIG = ROOT/'configs/m3w_moving_donor_augmentation_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
         ROOT/'src/world_model/m3w_donor_augmentation.py', ROOT/'tests/test_m3w_donor_augmentation.py')


def register():
    cfg = read(CONFIG)
    old.register()
    _, registered = cross.register()
    previous = read(old.PUBLIC/'analysis.json')
    analysis = read(cross.PUBLIC/'analysis.json')
    if not analysis['donor_bank_complete'] or not read(cross.PUBLIC/'replay.json')['six_predictions_exact']:
        raise ValueError('Complete verified producer-A bank required')
    if (cfg['arms'] != list(method.ARMS) or cfg['steps_per_fit'] != 4096 or cfg['fresh_models'] != 24
            or cfg['risk_budget'] != .02 or cfg['candidate_fractions'] != list(FRACTIONS)
            or cfg['augmentation_mass'] != .5 or cfg['minimum_cost_reduction'] != .01
            or cfg['minimum_passing_contexts'] != 2
            or any(cfg[k] for k in ('threshold_search', 'new_outer_C_H_readout', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed donor comparison and original rules required')
    contexts = read(ROOT/'outputs/publication_readiness_2026_09/moving_ego_frame_v1/registration_v2.json')['jobs']
    for c in contexts:
        group = set(c['producer_sites'])
        if group & (set(c['critic_sites']) | set(c['evaluation_sites'])):
            raise ValueError('Parent role overlap')
        for f in analysis['forecasts']:
            if f['job']['producer_fold'] == c['producer_fold']:
                if not set(f['job']['donor_sites']) <= group or set(f['job']['donor_sites']) & set(f['job']['fit_sites']):
                    raise ValueError('Invalid donor teacher ancestry')
    value = dict(config=cfg, settings=method.SETTINGS, jobs=contexts,
        files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        cached_crossfit=artifact(cross.PUBLIC/'analysis.json'),
        crossfit_registration=artifact(cross.PUBLIC/'registration.json'),
        prior_neural=artifact(old.PUBLIC/'analysis.json'),
        own_B_cache_refs=artifact(old.old.previous.audit.PUBLIC/'analysis.json'),
        parent_code={str(p.relative_to(ROOT)): sha(p) for p in
            (Path(method.parent.__file__), Path(method.relation.__file__), Path(method.neighbor.__file__),
             Path(method.causal_api.__file__), Path(method.causal_api.base.base.__file__))},
        result_role='already_exposed_own_B_source_held_training_donor_comparison',
        new_outer_C_H_readout=False, original_rules_unchanged=True, independent_confirmation=False)
    once(REG, stable(value))
    return cfg, value, previous


def committed():
    for p in (*FILES, REG):
        content = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
                                 capture_output=True, check=True).stdout
        if hashlib.sha256(content).hexdigest() != sha(p):
            raise ValueError('Commit exact experiment before data preparation or fitting')


def runtime(cfg, phase):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    start, last = time.monotonic(), [0.]
    def guard():
        now = time.monotonic()
        if now-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h limit; resume identical experiment')
        if now-last[0] > 1:
            last[0] = now
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
                raise OSError('Keep5GiB reserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['artifact_cap_bytes']:
                raise OSError('Private artifact cap exceeded')
    def beat(**values):
        guard()
        event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
        cross.atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    beat(state='ready', machine=platform.machine(), torch_threads=4, interop_threads=1, workers=0, device='cpu')
    return start, guard, beat


def prepare(reg, guard, beat):
    data = cross.load_data()
    bank = read(cross.PUBLIC/'analysis.json')
    receipts = []
    for fold in (0, 1):
        job = next(j for j in reg['jobs'] if j['producer_fold'] == fold)
        chunks = {k: [] for k in ('x', 'target', 'sites', 'query', 'ids')}
        candidates = unknown = identical = 0
        for forecast in bank['forecasts']:
            if forecast['job']['producer_fold'] != fold:
                continue
            with np.load(checked(forecast['forecast']), allow_pickle=False) as cache:
                ids, prediction = cache['local_ids'].copy(), cache['relative_prediction'].copy()
                np.testing.assert_array_equal(data['global_ids'][ids], cache['global_ids'])
            if not set(data['sites'][ids]) <= set(job['producer_sites']):
                raise ValueError('Wrong parent donor sources')
            for start in range(0, len(ids), 512):
                guard()
                ix = ids[start:start+512]
                neural = prediction[start:start+512].astype(float)
                origin = data['origin'][ix]
                physical = np.stack([baseline_numpy(data['history'][ix], k)-origin[:, None] for k in range(6)], 1)
                floor = physical[:, job['floor_index']]
                for action, alpha in enumerate(FRACTIONS, 1):
                    proposal = floor+alpha*(neural-floor) if alpha != 1 else neural
                    x, scale, allowed, paths = method.causal_features(data['geometry'][ix], data['shared_context'][ix],
                        physical, proposal, data['recordings'][ix], data['frames'][ix], data['global_ids'][ix],
                        data['sites'][ix], job['floor_index'])
                    y, known = method.supervision(paths, data['target_eval'][ix]-origin[:, None], data['valid'][ix],
                                                   scale, job['floor_index'], job['easy_cut'])
                    candidates += len(ix)
                    unknown += int((allowed & ~known).sum())
                    identical += int((~allowed).sum())
                    take = known & allowed
                    for k, v in dict(x=x, target=y, sites=data['sites'][ix], query=data['queries'][ix],
                                     ids=data['global_ids'][ix]*7+action).items():
                        chunks[k].append(v[take])
            beat(state='A_forecast_features_complete', producer_fold=fold, forecast=forecast['job']['key'])
        d = {k: np.concatenate(v) for k, v in chunks.items()}
        d['weights'] = method.parent.base.balanced_weights(d['sites'], d['query'])
        if not np.isfinite(d['x']).all() or not np.isfinite(d['target']).all() or len(set(d['ids'])) != len(d['ids']):
            raise ValueError('Finite unique known donor rows required')
        dest = PRIVATE/'donors'/f'A{fold}.npz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            with np.load(dest, allow_pickle=False) as saved:
                for k, v in d.items():
                    method.parent.storage.base.exact(saved[k], v)
        else:
            with dest.with_suffix('.pending').open('wb') as f:
                np.savez(f, **d)
            dest.with_suffix('.pending').replace(dest)
        receipts.append(dict(producer_fold=fold, cache=artifact(dest), rows=len(d['x']),
            arrays={k: method.parent.digest(v) for k, v in d.items()},
            sources=sorted(set(d['sites'])), candidates=candidates,
            unknown_legal_candidates=unknown, identical_to_floor_candidates=identical,
            easy_cut=job['easy_cut'], floor_index=job['floor_index'],
            teacher_exposure_excluded=True, source_annotation_causality_proven=False))
        beat(state='donor_bank_prepared', producer_fold=fold, rows=len(d['x']))
    once(PUBLIC/'donors.json', stable(dict(banks=receipts, result_source='fresh_run',
        features_before_outcome_filter=True, no_future_inference_input=True, unknowns_not_safe=True)))


def load_added(fold):
    record = next(b for b in read(PUBLIC/'donors.json')['banks'] if b['producer_fold'] == fold)
    with np.load(checked(record['cache']), allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files}
    for k, v in data.items():
        method.parent.storage.base.exact(method.parent.digest(v), record['arrays'][k])
    return data


def entries(reg, previous):
    for c in previous['contexts']:
        job = next(j for j in reg['jobs'] if j['key'] == c['job'])
        own, _ = old.load(c)
        added = load_added(job['producer_fold'])
        for f in c['folds']:
            for arm in method.ARMS:
                data = method.combined(own, added, f['source'], arm)
                identity = dict(registration_sha256=sha(REG), job=c['job'], source=f['source'], arm=arm,
                    train_ids=method.parent.digest(data['ids'][data['sites']!=f['source']]),
                    input_hash=method.parent.digest(data['x']), label_hash=method.parent.digest(data['target']),
                    weight_hash=method.parent.digest(data['weights']))
                yield c, f, arm, data, identity, own


def fit_path(c, f, arm, kind='fits'):
    return PRIVATE/kind/f'{c["job"]}_{f["source"]}_{arm}.pt.gz'


def train(reg, previous, guard, beat, resume=False, verify=False):
    refs = []
    for c, f, arm, data, identity, own in entries(reg, previous):
        path = fit_path(c, f, arm, 'verify' if verify else 'fits')
        state = method.fit(data, f['source'], identity, path, resume=resume, guard=guard,
                           heartbeat=lambda **kw: beat(job=c['job'], arm=arm, **kw))
        if not state['complete'] or state['step'] != 4096:
            raise ValueError('Full fixed budget required')
        if verify:
            method.exact(method.parent.storage.base.read_checkpoint(fit_path(c, f, arm)), state)
        else:
            out = PUBLIC/'fits'/f'{c["job"]}_{f["source"]}_{arm}.json'
            once(out, stable(dict(job=c['job'], source=f['source'], arm=arm,
                checkpoint=artifact(path), fit_rows=state['fit_rows'], step=state['step'],
                parameters=state['parameters'], trace=state['trace'], seconds=state['seconds'],
                fitting_sources=state['preprocess']['train_sources'],
                distinct_fitting_rows_drawn=int((state['visits'] > 0).sum()),
                result_source='fresh_run', evaluation_read=False)))
            refs.append(artifact(out))
        beat(state='risk_fit_verified' if verify else 'risk_fit_complete', job=c['job'], source=f['source'], arm=arm)
    if verify:
        once(PUBLIC/'verification.json', dict(all24_full_refits_exact=True, result_source='fresh_run'))
    else:
        once(PUBLIC/'training_freeze.json', dict(fits=refs, models=24, primary_updates=98304,
            all_before_final_readout=True, checkpoint_selection='final_only'))


def evaluate(reg, previous, beat, replay=False):
    freeze = read(PUBLIC/'training_freeze.json')
    if freeze['models'] != 24 or not read(PUBLIC/'verification.json')['all24_full_refits_exact']:
        raise ValueError('All fits and complete refits before final readout')
    results = []
    continued = read(ROOT/'outputs/publication_readiness_2026_09/moving_risk_convergence_v1/analysis.json')
    contexts = {}
    for c, f, arm, data, identity, own in entries(reg, previous):
        state = method.parent.storage.base.read_checkpoint(fit_path(c, f, arm))
        method.parent.storage.base.exact(state['identity'], identity)
        if not state['complete']:
            raise ValueError('Incomplete model')
        held = own['sites'] == f['source']
        pred = method.parent.predict(state, own['x'][held])
        metrics = method.parent.base.metrics(pred, own['target'][held], own['weights'][held], own['target_scale'][held][0])
        record = dict(job=c['job'], source=f['source'], arm=arm, rows=int(held.sum()),
            prediction_hash=method.parent.digest(pred), metrics=metrics,
            target_scale=own['target_scale'][held][0].tolist())
        resultpath = PRIVATE/'predictions'/f'{c["job"]}_{f["source"]}_{arm}.npy'
        if replay:
            np.testing.assert_array_equal(np.load(resultpath, allow_pickle=False), pred)
        else:
            resultpath.parent.mkdir(exist_ok=True)
            if resultpath.exists():
                np.testing.assert_array_equal(np.load(resultpath, allow_pickle=False), pred)
            else:
                with resultpath.open('wb') as out:
                    np.save(out, pred, allow_pickle=False)
        results.append(record)
        contexts.setdefault(c['job'], {}).setdefault(f['source'], {})[arm] = metrics
    summaries = []
    for c in previous['contexts']:
        means = {a: {k: float(np.mean([s[a][k] for s in contexts[c['job']].values()])) for k in
            ('normalized_MSE', 'total_joint_cost', 'false_safe_cost', 'false_reject_cost',
             'admitted_rows', 'false_safe_rows') if k in next(iter(contexts[c['job']].values()))[a]}
                 for a in method.ARMS}
        for name, key in (('shared64', 'shared64'), ('prior_neural', 'neural_plain')):
            means[name] = {k: float(np.mean([f['metrics'][key][k] for f in c['folds']])) for k in means['own_only']}
        old_c = next(x for x in continued['contexts'] if x['job']==c['job'])
        means['continued_neural'] = {k: float(np.mean([f['metrics']['long_plain'][k] for f in old_c['folds']])) for k in means['own_only']}
        own, aug, forest = (means[k] for k in ('own_only', 'augmented', 'shared64'))
        gain = 1-aug['total_joint_cost']/own['total_joint_cost'] if own['total_joint_cost'] else None
        summaries.append(dict(job=c['job'], source_macro=means, cost_reduction_vs_matched_own=gain,
            cost_reduction_vs_forest=1-aug['total_joint_cost']/forest['total_joint_cost'] if forest['total_joint_cost'] else None,
            passes_information_screen=bool(gain is not None and gain>=.01
                and aug['false_safe_cost']<=own['false_safe_cost'] and aug['normalized_MSE']<own['normalized_MSE']),
            safety_certificate=False))
    analysis = stable(dict(folds=results, contexts=summaries,
        passing_contexts=sum(c['passes_information_screen'] for c in summaries),
        result_source='fresh_run', original_rules_unchanged=True, identity_certification=False,
        new_outer_C_H_readout=False, original24group_primary='not_estimable_not_passed',
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    if replay:
        method.parent.storage.base.exact(read(PUBLIC/'analysis.json'), analysis)
    else:
        once(PUBLIC/'analysis.json', analysis)
    beat(state='evaluation_replayed' if replay else 'single_final_development_readout_complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register', 'prepare', 'pilot', 'train', 'verify', 'evaluate', 'replay'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, reg, previous = register()
        if args.phase == 'register':
            print('Registered24fixed-budget risk fits; original rules unchanged.')
            return
        committed()
        start, guard, beat = runtime(cfg, args.phase)
        if args.phase == 'prepare':
            prepare(reg, guard, beat)
        elif args.phase == 'pilot':
            entry = next(e for e in entries(reg, previous) if e[2]=='augmented')
            c, f, arm, d, identity, own = entry
            p, q = PRIVATE/'pilot/resumed.pt.gz', PRIVATE/'pilot/direct.pt.gz'
            if not p.exists():
                method.fit(d, f['source'], identity, p, stop_at=128, guard=guard, heartbeat=beat)
            a = method.fit(d, f['source'], identity, p, resume=True, stop_at=256, guard=guard, heartbeat=beat)
            b = method.fit(d, f['source'], identity, q, resume=args.resume, stop_at=256, guard=guard, heartbeat=beat)
            method.exact(a, b)
            once(PUBLIC/'pilot.json', stable(dict(resume_exact=True, steps=256, fit_rows=a['fit_rows'],
                seconds=a['seconds'], parameters=a['parameters'], checkpoint=artifact(p),
                projected24fit_seconds=a['seconds']/256*98304)))
        else:
            if not read(PUBLIC/'pilot.json')['resume_exact']:
                raise ValueError('Real resume pilot required')
            if args.phase in ('train', 'verify'):
                train(reg, previous, guard, beat, args.resume, args.phase=='verify')
            else:
                before = {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
                if args.phase=='replay':
                    def forbidden(*a, **kw):
                        raise RuntimeError('Fitting forbidden during replay')
                    method.fit = forbidden
                evaluate(reg, previous, beat, args.phase=='replay')
                if args.phase=='replay':
                    method.parent.storage.base.exact(before, {str(p):sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        out = (PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json')
        once(out, dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        beat(state='phase_complete')


if __name__ == '__main__':
    main()
