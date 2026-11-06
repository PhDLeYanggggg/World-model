"""Fixed source-held neural trajectory conditioning, without new outer readout."""
import argparse
import fcntl
import hashlib
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
    raise RuntimeError('Native arm64 required before numerical imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_producer_crossfit as source
from scripts import run_m3w_moving_prefix_actor as previous
from scripts import run_m3w_moving_visual_risk as vision
from src.world_model import m3w_visual_forecast as model_method
from src.training import m3w_visual_forecast as training
from scripts.run_m3w_moving_source_forecast import exact, same_scientific_checkpoint

np, torch = source.np, source.torch
read, sha, stable, once, checked, artifact = source.read, source.sha, source.stable, source.once, source.checked, source.artifact
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_visual_actor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_visual_actor_v1'
CONFIG = ROOT/'configs/m3w_moving_visual_actor_v1.json'; REG = PUBLIC/'registration_v2.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', Path(model_method.__file__), Path(training.__file__),
    ROOT/'tests/test_m3w_visual_forecast.py', ROOT/'tests/test_m3w_visual_actor_recovery.py', PUBLIC/'recovery_amendment.md')


def register():
    cfg = read(CONFIG); _, parent = source.register(); _, old = previous.register(); vision.register()
    preliminary = read(PUBLIC/'registration.json')
    for p in (CONFIG, PUBLIC/'protocol.md', Path(model_method.__file__), Path(training.__file__), ROOT/'tests/test_m3w_visual_forecast.py'):
        if sha(p) != preliminary['files'][str(p.relative_to(ROOT))]: raise ValueError('Recovery amendment cannot change the model or scientific definitions')
    if (cfg['treatment_arms'] != list(model_method.ARMS[1:]) or cfg['fresh_treatment_models'] != 12
            or cfg['full_treatment_refits'] != 2 or cfg['full_raw_control_refits'] != 1
            or cfg['cached_raw_controls'] != 6 or cfg['steps_per_model'] != parent['training']['steps']
            or cfg['steps_per_model'] != 4000 or cfg['forecaster_seeds'] != [17]
            or cfg['context_columns'] != 46 or cfg['original_risk_budget'] != .02
            or any(cfg[k] for k in ('warm_start', 'threshold_search', 'new_own_B_readout', 'new_outer_C_H_readout',
                'identity_certification', 'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Full matched A-only actor component experiment and original rules required')
    controls = read(source.PUBLIC/'analysis.json')['forecasts']
    if len(controls) != 6 or sum(c['rows'] for c in controls) != 125540:
        raise ValueError('Six complete cached control banks required')
    for c in controls: checked(c['checkpoint']); checked(c['forecast'])
    dependencies = (Path(source.__file__), Path(previous.__file__), ROOT/'src/world_model/m3w_native_forecast.py',
        ROOT/'src/world_model/m3w_european_source_forecast.py', ROOT/'src/world_model/m3w_context_conditioning.py',
        ROOT/'src/world_model/m3w_eqmotion_adapter.py', ROOT/'src/world_model/m3w_supervised_intervention.py')
    value = stable(dict(config=cfg, architecture=parent['architecture'], training=parent['training'],
        jobs=parent['jobs'], controls=controls, cuts=old['cuts'], author_core=parent['author_core'],
        files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        parents=[artifact(p) for p in (source.PUBLIC/'registration.json', source.PUBLIC/'population.json',
            source.PUBLIC/'analysis.json', previous.PUBLIC/'analysis.json', vision.REG,
            vision.PUBLIC/'context.json', vision.PUBLIC/'analysis.json', vision.PUBLIC/'calibration_analysis.json')],
        measured_inputs=vision.measurement_bundle(), torch=str(torch.__version__), numpy=np.__version__,
        preliminary_registration=artifact(PUBLIC/'registration.json'), definition_version=2, recovery_only_amendment=True,
        machine=platform.machine(), role='producer_A_component_not_independent_confirmation',
        original_rules_unchanged=True, new_own_B_readout=False, new_outer_C_H_readout=False))
    once(REG, value)
    once(PUBLIC/'input_admission_v2.json', dict(registration_sha256=sha(REG),
        version='past_image_derived_actor_conditioning_v1', original_admission_files_edited=False,
        role='supervised_training_existing_producer_A_only', context_columns=46,
        candidate_specific_rollout_context=False, future_targets_as_inputs=False,
        future_masks_as_inputs=False, future_images_as_inputs=False,
        original_risk_support_NA_rules_unchanged=True, new_own_B_readout=False, new_outer_C_H_readout=False))
    return cfg, value


def committed(reg):
    for p in (*FILES, REG, PUBLIC/'input_admission_v2.json'):
        blob = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != sha(p): raise ValueError('Commit the exact experiment before real preparation/fitting')
    for p, digest in reg['dependencies'].items():
        if sha(ROOT/p) != digest: raise ValueError('Changed parent implementation')


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    start, last = time.monotonic(), [0.]
    def guard():
        now = time.monotonic()
        if now-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('Resume unchanged study after12h boundary')
        if now-last[0] > 1:
            last[0] = now
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Keep5GiBreserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['artifact_cap_bytes']:
                raise OSError('New private artifact cap exceeded')
    def beat(**values):
        guard(); event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **values)
        source.atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(source.json.dumps(event)+'\n')
        print(source.json.dumps(event), flush=True)
    beat(state='ready', machine=platform.machine(), device='cpu', threads=4, interop=1, workers=0, resource_probing=False)
    return start, guard, beat


def atomic_array(path, value):
    if path.exists():
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)
        return
    path.parent.mkdir(parents=True, exist_ok=True); pending = path.with_suffix('.pending')
    with pending.open('wb') as f:
        np.save(f, value, allow_pickle=False); f.flush(); os.fsync(f.fileno())
    pending.replace(path)


def completed_fit(path, expected_identity):
    if not path.exists(): return None
    receipt = read(path)
    state = torch.load(checked(receipt['checkpoint']), map_location='cpu', weights_only=False)
    if (receipt.get('identity') != expected_identity or state['identity'] != expected_identity
            or state['image_arm'] != expected_identity['arm'] or state['step'] != 4000
            or not receipt['result']['complete'] or receipt['result']['step'] != 4000):
        raise ValueError('Completed fit must retain the exact full-budget identity')
    return receipt


def prepare(beat):
    past = vision.images.past_arrays(read(vision.images.CONFIG)); ids = past['global_ids']
    n = len(ids); tables = {name: np.empty((n, 46), np.float32) for name in ('quality', 'motion')}
    filled = np.zeros(n, bool); order = np.argsort(ids)
    for fold in (0, 1):
        ref = next(r['cache'] for r in read(vision.PUBLIC/'context.json')['banks'] if r['bank'] == f'A{fold}')
        table = vision.parent.load_cache(ref); keys = table['ids']//7
        unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        pos = np.searchsorted(ids[order], unique)
        if (pos == n).any(): raise ValueError('Missing exact A row context')
        ix = order[pos]; np.testing.assert_array_equal(ids[ix], unique)
        if filled[ix].any(): raise ValueError('Overlapping producer supports')
        for name in tables:
            value = np.c_[table[name][:, :40], table[name][:, 48:54]]
            np.testing.assert_array_equal(value, value[first][inverse])
            tables[name][ix] = value[first]
        filled[ix] = True
        beat(state='causal_context_joined', producer_fold=fold, rows=len(unique), target_arrays_opened=False)
    if not filled.all() or len(set(past['sites'])) != 14 or len(set(past['recordings'])) != 34:
        raise ValueError('Retain all125540Arows and all source/recording groups')
    files = {}
    for name, value in dict(**tables, global_ids=np.asarray(ids)).items():
        p = PRIVATE/'arrays'/f'{name}.npy'; atomic_array(p, value)
        files[name] = artifact(p)
    once(PUBLIC/'population.json', dict(files=files, rows=n, columns=46, sources=14, recordings=34,
        exact_original_global_ID_joins=True, all_candidate_copies_action_invariant=True,
        target_arrays_opened=False, preparation_role='past_input_only_cached_verified_measurements'))


def data_and_jobs():
    data = source.load_data(); meta = read(PUBLIC/'population.json')
    for k, ref in meta['files'].items():
        values = np.load(checked(ref), mmap_mode='r', allow_pickle=False)
        if k == 'global_ids': np.testing.assert_array_equal(data[k], values)
        else: data[k] = values
    jobs = source.designs(data, read(source.PUBLIC/'registration.json'))
    if len(jobs) != 6: raise ValueError('Full source roster required')
    return data, jobs


def identity(entry, arm):
    return dict(parent_identity=entry[3], registration_sha256=sha(REG), arm=arm, loss='original_available12_ADE')


def fit_one(data, entry, reg, home, arm, guard, beat, resume=False, stop_at=None):
    pr = (dict(mean=np.zeros(46), std=np.ones(46)) if arm == 'raw_control'
        else model_method.normalizer(data['quality' if arm == 'quality_geometry' else 'motion'], entry[1]))
    torch.manual_seed(17)
    model = model_method.VisualSourceForecaster(reg['architecture'], entry[1]['baseline_index'], pr['mean'], pr['std'], arm)
    result = training.fit(model, data, entry[1], arm=arm, seed=17, settings=reg['training'],
        identity=identity(entry, arm), directory=home, resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **kw: beat(job=entry[0]['key'], arm=arm, **kw))
    state = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
    if state['draws'][entry[1]['held_ids']].any() or result['parameters'] != 3028996:
        raise ValueError('Held-source exposure or changed actor capacity')
    return result, state


def pilot(cfg, reg, data, jobs, guard, beat):
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot is immutable')
    p, q = PRIVATE/'pilot/resumed', PRIVATE/'pilot/direct'
    if not (p/'checkpoint.pt').exists(): fit_one(data, jobs[0], reg, p, 'motion_geometry', guard, beat, stop_at=100)
    result, a = fit_one(data, jobs[0], reg, p, 'motion_geometry', guard, beat, resume=True, stop_at=200)
    direct, b = fit_one(data, jobs[0], reg, q, 'motion_geometry', guard, beat, resume=(q/'checkpoint.pt').exists(), stop_at=200)
    same_scientific_checkpoint(a, b)
    once(PUBLIC/'pilot.json', stable(dict(result_source='fresh_run', actual_Torch_training=True,
        resume100plus100_equals200=True, all_scientific_state_exact=True, parameters=result['parameters'],
        prefix=artifact(p/'checkpoint.pt'), direct200_seconds=direct['seconds'],
        projected15_full_fits_seconds=direct['seconds']/200*60000,
        pilot_is_not_scientific_success=True, new_own_B_readout=False)))


def control(reg, data, jobs, guard, beat, resume):
    if completed_fit(PUBLIC/'raw_control_refit.json', identity(jobs[0], 'raw_control')) is not None:
        beat(state='completed_raw_control_verified'); return
    home = PRIVATE/'raw_control'; result, new = fit_one(data, jobs[0], reg, home, 'raw_control', guard, beat, resume)
    ref = next(c for c in reg['controls'] if c['job']['key'] == jobs[0][0]['key'])
    old = torch.load(checked(ref['checkpoint']), map_location='cpu', weights_only=False)
    for k, v in old['model'].items(): exact(v, new['model'][k])
    for k in ('draws', 'train_ids', 'factors', 'sampler_rng', 'torch_rng', 'seed', 'step', 'settings'): exact(old[k], new[k])
    exact(old['optimizer']['state'], new['optimizer']['state'])
    for x, y in zip(old['optimizer']['param_groups'], new['optimizer']['param_groups']):
        exact({k:v for k,v in x.items() if k != 'params'}, {k:v for k,v in y.items() if k != 'params'})
        exact(x['params'], y['params'][:len(x['params'])])
        if len(y['params']) != len(x['params'])+2: raise ValueError('Only two inactive context parameters added')
    a, b = ({r['step']: r for r in s['losses']} for s in (old, new))
    if set(b)-set(a) or set(a)-set(b)-{20, 40}: raise ValueError('Unexpected raw control trace change')
    for step in b: exact(a[step], b[step])
    if not result['complete']: raise ValueError('Full4000update old control required')
    once(PUBLIC/'raw_control_refit.json', stable(dict(result_source='fresh_run', result=result,
        identity=identity(jobs[0], 'raw_control'),
        checkpoint=artifact(home/'checkpoint.pt'), shared_core_optimizer_sampler_RNG_and_common_trace_exact=True,
        inactive_context_parameters=1728, old_pilot_only_log_steps=sorted(set(a)-set(b)))))


def train(cfg, reg, data, jobs, guard, beat, resume, verify=False):
    if not read(PUBLIC/'raw_control_refit.json')['shared_core_optimizer_sampler_RNG_and_common_trace_exact']:
        raise ValueError('Verify original control before treatment matrix')
    prefix = read(PUBLIC/'pilot.json')['prefix']; receipts = []
    for i, entry in enumerate(jobs[:1] if verify else jobs):
        for arm in cfg['treatment_arms']:
            key = entry[0]['key']+'_'+arm; home = PRIVATE/('verify' if verify else 'fits')/key
            if not verify:
                done = completed_fit(PUBLIC/'fits'/f'{key}.json', identity(entry, arm))
                if done is not None:
                    receipts.append(dict(key=key, checkpoint=done['checkpoint']))
                    beat(state='completed_fit_verified', complete=len(receipts), total=12)
                    continue
            if not verify and i == 0 and arm == 'motion_geometry' and not (home/'checkpoint.pt').exists():
                home.mkdir(parents=True, exist_ok=True); shutil.copyfile(checked(prefix), home/'checkpoint.pt')
            result, state = fit_one(data, entry, reg, home, arm, guard, beat, resume or (home/'checkpoint.pt').exists())
            if not result['complete'] or state['step'] != 4000: raise ValueError('Complete fixed budget required')
            if verify:
                original = torch.load(checked(read(PUBLIC/'fits'/f'{key}.json')['checkpoint']), map_location='cpu', weights_only=False)
                same_scientific_checkpoint(original, state)
            else:
                raw = torch.load(checked(reg['controls'][i]['checkpoint']), map_location='cpu', weights_only=False)
                for k in ('draws', 'factors', 'train_ids', 'sampler_rng', 'torch_rng'): exact(raw[k], state[k])
                once(PUBLIC/'fits'/f'{key}.json', stable(dict(result_source='fresh_run', job=entry[0], arm=arm,
                    identity=identity(entry, arm), checkpoint=artifact(home/'checkpoint.pt'), result=result,
                    raw_control_draws_and_factors_exact=True)))
            receipts.append(dict(key=key, checkpoint=artifact(home/'checkpoint.pt')))
            beat(state='complete_refit_exact' if verify else 'fit_complete', complete=len(receipts), total=2 if verify else 12)
    once(PUBLIC/('verification.json' if verify else 'training.json'), dict(models=len(receipts),
        updates=len(receipts)*4000, full_refits_exact=verify, first_fixed_source_design_only=verify, receipts=receipts))


def forecasts(reg, data, jobs, beat, replay=False):
    if not read(PUBLIC/'verification.json')['full_refits_exact']: raise ValueError('Both full treatment refits required')
    refs = []
    for entry in jobs:
        ix = entry[2]
        for arm in reg['config']['treatment_arms']:
            key = entry[0]['key']+'_'+arm; fit = read(PUBLIC/'fits'/f'{key}.json')
            state = torch.load(checked(fit['checkpoint']), map_location='cpu', weights_only=False)
            if state['identity'] != identity(entry, arm) or state['step'] != 4000: raise ValueError('Frozen final actor only')
            torch.manual_seed(17)
            model = model_method.VisualSourceForecaster(reg['architecture'], entry[1]['baseline_index'],
                state['model']['image_mean'].numpy(), state['model']['image_std'].numpy(), arm)
            model.load_state_dict(state['model']); model.eval(); chunks = []
            with torch.no_grad():
                for begin in range(0, len(ix), 128):
                    ids = ix[begin:begin+128]
                    inp = model_method.pack_inputs(data['geometry'][ids], data['quality' if arm == 'quality_geometry' else 'motion'][ids])
                    chunks.append(model(inp).numpy())
                    if begin % 4096 == 0: beat(state='predicting_A_donors_only', job=key, rows=min(begin+128, len(ix)))
            prediction = np.concatenate(chunks)
            if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite actor forecast')
            p = PRIVATE/'forecasts'/f'{key}.npz'; out = PUBLIC/'forecasts'/f'{key}.json'
            if p.exists() or out.exists():
                with np.load(checked(read(out)['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], data['global_ids'][ix]); np.testing.assert_array_equal(z['prediction'], prediction)
            else:
                if replay: raise ValueError('Replay requires existing complete prediction banks')
                vision.parent.full.save_arrays(p, dict(prediction=prediction, ids=data['global_ids'][ix]))
                once(out, dict(job=entry[0]['key'], arm=arm, rows=len(ix), forecast=artifact(p)))
            refs.append(artifact(out))
    if len(refs) != 12: raise ValueError('All twelve banks required')
    if not replay: once(PUBLIC/'forecast_freeze.json', dict(banks=refs, all_predictions_frozen_before_scoring=True))


def evaluate(reg, data, jobs, beat, replay=False):
    if not read(PUBLIC/'forecast_freeze.json')['all_predictions_frozen_before_scoring']: raise ValueError('Freeze all actors before scoring')
    rows = []; records = []
    for i, entry in enumerate(jobs):
        ix = entry[2]; key = entry[0]['key']; predictions = {}
        with np.load(checked(reg['controls'][i]['forecast']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['global_ids'], data['global_ids'][ix]); predictions['raw'] = z['relative_prediction'].astype(float)
        for arm in reg['config']['treatment_arms']:
            ref = read(PUBLIC/'forecasts'/f'{key}_{arm}.json')
            with np.load(checked(ref['forecast']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], data['global_ids'][ix]); predictions[arm] = z['prediction'].astype(float)
        floor = previous.baseline_numpy(data['history'][ix], 1)-data['origin'][ix, None]
        target = data['target_eval'][ix]-data['origin'][ix, None]; valid = data['valid'][ix]
        cut = reg['cuts'][str(entry[0]['producer_fold'])]
        for contrast, first, second in (('primary', 'quality_geometry', 'motion_geometry'),
                ('quality_bundle', 'raw', 'quality_geometry'), ('motion_bundle', 'raw', 'motion_geometry')):
            for grouping, dest, name in ((data['sites'][ix], rows, 'source'), (data['recordings'][ix], records, 'recording')):
                for group in sorted(set(grouping)):
                    at = grouping == group; _, inverse, count = np.unique(data['queries'][ix][at], return_inverse=True, return_counts=True)
                    w = 1./count[inverse]; w /= w.sum()
                    metrics = previous.statistics(predictions[first][at], predictions[second][at], floor[at], target[at], valid[at],
                        w, cut['easy'], cut['hard'])
                    dest.append(dict(job=key, producer_fold=entry[0]['producer_fold'], contrast=contrast,
                        **{name: str(group)}, summary=metrics))
        beat(state='source_A_paired_readout', job=key)
    primary = [r for r in rows if r['contrast'] == 'primary']
    if len(primary) != 14 or sum(r['summary'][-1]['rows'] for r in primary) != 125540:
        raise ValueError('Complete source population required')
    lower = np.array([r['summary'][-1]['paired_completion_lower_pp'] for r in primary])
    known = np.array([r['summary'][-1]['paired_known_pp'] for r in primary])
    draws = np.random.default_rng(reg['config']['bootstrap_seed']).integers(0, 14, (3000, 14))
    ci = np.quantile(lower[draws].mean(1), [.025, .975])
    folds = {str(f): float(np.mean([r['summary'][-1]['paired_known_pp'] for r in primary if r['producer_fold'] == f])) for f in (0, 1)}
    screen = bool(lower.mean() > 0 and ci[0] > 0 and (known > 0).sum() >= 8 and min(folds.values()) > 0)
    result = stable(dict(result_source='fresh_run', sources=rows, recordings=records, rows=125540,
        primary=dict(known_pp=float(known.mean()), completion_lower_pp=float(lower.mean()), interval95=ci.tolist(),
            positive_known_sources=int((known > 0).sum()), fold_known_means=folds, component_progression_screen_pass=screen),
        one_forecaster_seed=True, exploratory_component_only=True, independent_confirmation=False,
        original_rules_unchanged=True, new_controller_training=False, new_own_B_readout=False, new_outer_C_H_readout=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if replay: exact(read(PUBLIC/'analysis.json'), result)
    else: once(PUBLIC/'analysis.json', result)


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('register', 'prepare', 'pilot', 'control', 'train', 'verify', 'forecast', 'evaluate', 'replay'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, reg = register()
        if args.phase == 'register': print('Registered twelve full image-conditioned actors,one old control,two full refits;original rules.'); return
        committed(reg); start, guard, beat = runtime(cfg, args.phase)
        before = {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase == 'prepare': prepare(beat)
        else:
            data, jobs = data_and_jobs()
            if args.phase == 'pilot': pilot(cfg, reg, data, jobs, guard, beat)
            elif args.phase == 'control': control(reg, data, jobs, guard, beat, args.resume)
            elif args.phase in ('train', 'verify'): train(cfg, reg, data, jobs, guard, beat, args.resume, args.phase == 'verify')
            elif args.phase == 'forecast': forecasts(reg, data, jobs, beat)
            elif args.phase == 'evaluate': evaluate(reg, data, jobs, beat)
            else:
                forecasts(reg, data, jobs, beat, replay=True); evaluate(reg, data, jobs, beat, replay=True)
                exact(before, {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        dest = PRIVATE/'replays' if args.phase == 'replay' else PUBLIC/'executions'
        once(dest/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json', dict(pid=os.getpid(), phase=args.phase,
            seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, actual_Torch=True))
        beat(state='phase_complete')


if __name__ == '__main__': main()
