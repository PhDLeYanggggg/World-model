"""Source-held image-coordinate geometry repair, without a new outer readout."""
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
from scripts import run_m3w_moving_prefix_actor as metrics
from scripts import run_m3w_moving_past_background as images
from scripts.run_m3w_moving_source_forecast import exact, same_scientific_checkpoint
from src.world_model import m3w_camera_frame as method
from src.training import m3w_camera_frame as training
from src.world_model.m3w_european_source_forecast import BASELINES

np, torch = source.np, source.torch
read, sha, stable, once, checked, artifact = source.read, source.sha, source.stable, source.once, source.checked, source.artifact
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_camera_frame_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_camera_frame_v1'
CONFIG = ROOT/'configs/m3w_moving_camera_frame_v1.json'; REG = PUBLIC/'registration_v2.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', Path(method.__file__), Path(training.__file__),
    ROOT/'tests/test_m3w_camera_frame.py', PUBLIC/'preflight_amendment.md')


def pair_binding():
    digest = hashlib.sha256(); count = 0
    for p in sorted((images.PRIVATE/'pairs').glob('*/*.json')):
        value = read(p)
        if value['registration_sha256'] != sha(images.REG):
            raise ValueError('Changed past-pair registration')
        digest.update(str(p.relative_to(ROOT)).encode()); digest.update(sha(p).encode()); count += 1
    if count != 1817: raise ValueError('Complete previously measured A-side pairs required')
    return dict(sha256=digest.hexdigest(), pairs=count, result_source='cached_verified')


def register():
    cfg = read(CONFIG); _, parent = source.register(); _, old = metrics.register(); images.register()
    draft = read(PUBLIC/'registration.json')
    for path in (CONFIG, PUBLIC/'protocol.md', Path(method.__file__), Path(training.__file__)):
        if sha(path) != draft['files'][str(path.relative_to(ROOT))]:
            raise ValueError('Preflight guards must not change the model or scientific definitions')
    if (cfg['fresh_treatment_models'] != 6 or cfg['forecaster_seeds'] != [17]
            or cfg['steps_per_model'] != parent['training']['steps'] or cfg['steps_per_model'] != 4000
            or cfg['original_risk_budget'] != .02 or cfg['expected_rows'] != 125540
            or any(cfg[k] for k in ('warm_start', 'threshold_search', 'identity_certification',
                'new_own_B_readout', 'new_outer_C_H_readout', 'deployment_changed', 'submission_ready',
                'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed full-budget A-only component and original risk rules required')
    controls = read(source.PUBLIC/'analysis.json')['forecasts']
    if len(controls) != 6 or sum(c['rows'] for c in controls) != 125540:
        raise ValueError('Retain all six complete cached controls')
    for c in controls: checked(c['checkpoint']); checked(c['forecast'])
    dependencies = (Path(source.__file__), Path(metrics.__file__), Path(images.__file__),
        ROOT/'src/world_model/m3w_native_forecast.py', ROOT/'src/world_model/m3w_context_conditioning.py',
        ROOT/'src/world_model/m3w_european_source_forecast.py', ROOT/'src/world_model/m3w_eqmotion_adapter.py',
        ROOT/'src/world_model/m3w_supervised_intervention.py')
    reg = stable(dict(config=cfg, training=parent['training'], architecture=parent['architecture'],
        jobs=parent['jobs'], cuts=old['cuts'], controls=controls, author_core=parent['author_core'],
        files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        parents=[artifact(p) for p in (source.PUBLIC/'registration.json', source.PUBLIC/'population.json',
            source.PUBLIC/'analysis.json', metrics.PUBLIC/'registration.json', images.REG, images.PUBLIC/'measurement.json',
            ROOT/'outputs/publication_readiness_2026_09/moving_visual_actor_v1/analysis.json')],
        past_pair_binding=pair_binding(), torch=str(torch.__version__), numpy=np.__version__,
        pre_execution_draft=artifact(PUBLIC/'registration.json'), definition_version=2,
        pre_fitting_recovery_guards_only=True,
        original_rules_unchanged=True, independent_confirmation=False, new_own_B_readout=False,
        new_outer_C_H_readout=False, role='exposed_producer_A_component_not_independent_test'))
    once(REG, reg)
    once(PUBLIC/'input_admission_v2.json', dict(registration_sha256=sha(REG),
        version='past_image_native_query_view_geometry_v1', previous_admission_files_edited=False,
        role='supervised_training_existing_producer_A_only',
        observed_inputs=['past_agent_neighbor_geometry', 'past_image_affine_proxy'],
        predicted_camera_operator_is_past_only_extrapolation=True, observed_future_camera_input=False,
        future_target_or_mask_in_model_input=False, test_endpoint_goals=False,
        original_risk_support_NA_rules_unchanged=True, identity_certification=False))
    return cfg, reg


def committed(reg):
    for p in (*FILES, REG, PUBLIC/'input_admission_v2.json'):
        blob = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != sha(p): raise ValueError('Commit exact experiment before real preparation/fitting')
    for p, digest in reg['dependencies'].items():
        if sha(ROOT/p) != digest: raise ValueError('Changed parent code')


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged checkpoint')))
    start, last = time.monotonic(), [0.]
    def guard():
        now = time.monotonic()
        if now-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary;resume unchanged model')
        if now-last[0] > 1:
            last[0] = now
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Keep2GiBreserve')
            if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['artifact_cap_bytes']:
                raise OSError('New private artifact cap exceeded')
    def beat(**values):
        if not str(values.get('state', '')).startswith('interrupted'): guard()
        event = dict(pid=os.getpid(), phase=phase, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **values)
        source.atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(source.json.dumps(event)+'\n')
        print(source.json.dumps(event), flush=True)
    beat(state='ready', machine=platform.machine(), device='cpu', threads=4, interop=1, workers=0, resource_probing=False)
    return start, guard, beat


def atomic_array(path, value):
    if path.exists():
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), value); return
    path.parent.mkdir(parents=True, exist_ok=True); pending = path.with_suffix('.pending')
    with pending.open('wb') as f:
        np.save(f, value, allow_pickle=False); f.flush(); os.fsync(f.fileno())
    pending.replace(path)


def prepare(cfg, beat):
    past = images.past_arrays(read(images.CONFIG))
    source_population = read(source.PUBLIC/'population.json')
    geometry = np.load(checked(source_population['files']['geometry']), mmap_mode='r', allow_pickle=False)
    np.testing.assert_array_equal(geometry[:, :16], (past['history']-past['history'][:, -1, None]).astype(np.float32).reshape(-1, 16))
    mask = geometry[:, 230:294].reshape(-1, 8, 8).astype(bool)
    times = geometry[:, 166:230].reshape(-1, 8, 8)
    expected = np.broadcast_to(geometry[:, None, 16:24], times.shape)
    if not np.isclose(times[mask], expected[mask], atol=1e-6, rtol=1e-6).all():
        raise ValueError('Complete past operator grid must align with observed neighbor times')
    query, row_index = np.unique(np.c_[past['recordings'], past['frames']], axis=0, return_inverse=True)
    if (len(query) != cfg['expected_queries'] or len(past['global_ids']) != cfg['expected_rows']
            or len(set(past['sites'])) != 14 or len(set(past['recordings'])) != 34):
        raise ValueError('Complete original past population required')
    names = sorted(read(ROOT/read(images.CONFIG)['admission'])['recordings'])
    history, prediction, active, support, roster = [], [], [], [], []
    pair_cache = {}
    for qi, (recording, frame) in enumerate(query):
        node, frame = names[int(recording)], int(frame); flows = {}
        for f in range(frame-6, frame+1):
            if (node, f) not in pair_cache:
                p = images.PRIVATE/'pairs'/node/f'{f:06d}.json'; value = read(p)
                if value['node'] != node or value['frame'] != f or value['registration_sha256'] != sha(images.REG):
                    raise ValueError('Changed past-pair metadata')
                pair_cache[node, f] = value['measurement']
            flows[f] = pair_cache[node, f]
        maps = method.query_maps(flows, frame)
        history.append(maps['past']); prediction.append(maps['prediction'])
        active.append(maps['active']); support.append(maps['supported_pairs'])
        roster.append(dict(node=node, frame=frame, rows=int((row_index == qi).sum()),
            supported_pairs=maps['supported_pairs'], active=maps['active']))
    arrays = dict(camera_past=np.array(history), camera_prediction=np.array(prediction),
        camera_query_index=row_index.astype(np.int32), camera_active=np.array(active, bool),
        global_ids=np.asarray(past['global_ids']))
    refs = {}
    for k, v in arrays.items():
        p = PRIVATE/'arrays'/f'{k}.npy'; atomic_array(p, v); refs[k] = artifact(p)
    once(PUBLIC/'population.json', dict(result_source='fresh_causal_composition_cached_verified_measurements',
        files=refs, rows=len(row_index), sources=14, recordings=34, queries=len(query),
        active_queries=sum(active), active_rows=int(np.array(active)[row_index].sum()),
        roster=roster, target_arrays_opened=False, no_row_deletion=True,
        future_images_or_poses_used=False, geometry_and_history_exactly_aligned=True,
        observed_neighbor_times_aligned=True, coordinate_unit='pixel', metric_status='unverified_image_coordinates',
        observation_steps=8, prediction_steps=12, annotation_stride=1, effective_seconds_verified=False,
        pair_binding=pair_binding()))
    beat(state='all_past_camera_operators_prepared', rows=len(row_index), active_queries=sum(active), targets_opened=False)


def data_and_jobs():
    data = source.load_data(); meta = read(PUBLIC/'population.json')
    for k, ref in meta['files'].items():
        array = np.load(checked(ref), mmap_mode='r', allow_pickle=False)
        if k == 'global_ids': np.testing.assert_array_equal(array, data[k])
        else: data[k] = array
    jobs = source.designs(data, read(source.PUBLIC/'registration.json'))
    if len(jobs) != 6: raise ValueError('Complete six-design source roster required')
    return data, jobs


def identity(entry, arm):
    return dict(parent_identity=entry[3], camera_arm=arm, registration_sha256=sha(REG))


def fit_one(data, entry, reg, home, arm, guard, beat, resume=False, stop_at=None):
    torch.manual_seed(17); model = method.CameraFrameForecaster(reg['architecture'], entry[1]['baseline_index'], arm)
    result = training.fit(model, data, entry[1], arm=arm, seed=17, settings=reg['training'],
        identity=identity(entry, arm), directory=home, resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **kw: beat(job=entry[0]['key'], arm=arm, **kw))
    state = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
    if result['parameters'] != 3027268 or state['draws'][entry[1]['held_ids']].any():
        raise ValueError('Changed capacity or held-source exposure')
    return result, state


def complete_receipt(path, expected):
    if not path.exists(): return None
    receipt = read(path); state = torch.load(checked(receipt['checkpoint']), map_location='cpu', weights_only=False)
    if (receipt['identity'] != expected or state['identity'] != expected or state['step'] != 4000
            or receipt['result']['step'] != 4000 or not receipt['result']['complete']):
        raise ValueError('Changed completed-model identity')
    return receipt


def pilot(reg, data, jobs, guard, beat):
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot is immutable')
    a, b = PRIVATE/'pilot/resumed', PRIVATE/'pilot/direct'
    if not (a/'checkpoint.pt').exists(): fit_one(data, jobs[0], reg, a, 'camera_frame', guard, beat, stop_at=100)
    result, left = fit_one(data, jobs[0], reg, a, 'camera_frame', guard, beat, resume=True, stop_at=200)
    direct, right = fit_one(data, jobs[0], reg, b, 'camera_frame', guard, beat,
        resume=(b/'checkpoint.pt').exists(), stop_at=200)
    same_scientific_checkpoint(left, right)
    once(PUBLIC/'pilot.json', stable(dict(result_source='fresh_run', actual_Torch=True,
        real100plus100_equals200=True, full_scientific_state_exact=True, prefix=artifact(a/'checkpoint.pt'),
        direct200_seconds=direct['seconds'], projected8_full_fits_seconds=direct['seconds']/200*32000,
        nominal_parameters=result['parameters'], trainable_parameters=result['trainable_parameters'],
        pilot_not_scientific_success=True)))


def control(reg, data, jobs, guard, beat, resume):
    if complete_receipt(PUBLIC/'control.json', identity(jobs[0], 'identity')) is not None: return
    home = PRIVATE/'control'; result, new = fit_one(data, jobs[0], reg, home, 'identity', guard, beat, resume)
    old = torch.load(checked(reg['controls'][0]['checkpoint']), map_location='cpu', weights_only=False)
    for k in ('model', 'optimizer', 'draws', 'train_ids', 'factors', 'sampler_rng', 'torch_rng', 'seed', 'step', 'settings'):
        exact(old[k], new[k])
    x, y = ({r['step']: r for r in value['losses']} for value in (old, new))
    if set(y)-set(x) or set(x)-set(y)-{20, 40}: raise ValueError('Changed common control training trace')
    for step in y: exact(x[step], y[step])
    once(PUBLIC/'control.json', dict(identity=identity(jobs[0], 'identity'), result=result,
        checkpoint=artifact(home/'checkpoint.pt'), complete4000shared_state_and_common_trace_exact=True))


def train(reg, data, jobs, guard, beat, resume, verify=False):
    if not read(PUBLIC/'control.json')['complete4000shared_state_and_common_trace_exact']:
        raise ValueError('Full original-control parity required first')
    receipts = []; prefix = read(PUBLIC/'pilot.json')['prefix']
    for i, entry in enumerate(jobs[:1] if verify else jobs):
        key = entry[0]['key']; home = PRIVATE/('verify' if verify else 'fits')/key
        done = None if verify else complete_receipt(PUBLIC/'fits'/f'{key}.json', identity(entry, 'camera_frame'))
        if done is not None:
            receipts.append(done['checkpoint']); beat(state='completed_fit_verified', complete=len(receipts)); continue
        if not verify and i == 0 and not (home/'checkpoint.pt').exists():
            home.mkdir(parents=True, exist_ok=True); shutil.copyfile(checked(prefix), home/'checkpoint.pt')
        result, state = fit_one(data, entry, reg, home, 'camera_frame', guard, beat,
            resume or (home/'checkpoint.pt').exists())
        if not result['complete']: raise ValueError('Full4000update budget required')
        if verify:
            original = torch.load(checked(read(PUBLIC/'fits'/f'{key}.json')['checkpoint']), map_location='cpu', weights_only=False)
            same_scientific_checkpoint(original, state)
        else:
            old = torch.load(checked(reg['controls'][i]['checkpoint']), map_location='cpu', weights_only=False)
            for k in ('draws', 'factors', 'train_ids', 'sampler_rng', 'torch_rng'): exact(old[k], state[k])
            once(PUBLIC/'fits'/f'{key}.json', stable(dict(result_source='fresh_run', job=entry[0], result=result,
                identity=identity(entry, 'camera_frame'), checkpoint=artifact(home/'checkpoint.pt'),
                original_draws_factors_and_RNG_exact=True)))
        receipts.append(artifact(home/'checkpoint.pt')); beat(state='full_refit_exact' if verify else 'fit_complete', complete=len(receipts))
    once(PUBLIC/('verification.json' if verify else 'training.json'), dict(models=len(receipts),
        steps_each=4000, full_refits_exact=verify, first_design_only=verify, checkpoints=receipts))


def forecasts(reg, data, jobs, beat, replay=False):
    if not read(PUBLIC/'verification.json')['full_refits_exact']: raise ValueError('Full treatment refit required')
    refs = []
    for entry in jobs:
        key = entry[0]['key']; ids = entry[2]
        fit = read(PUBLIC/'fits'/f'{key}.json'); state = torch.load(checked(fit['checkpoint']), map_location='cpu', weights_only=False)
        if state['identity'] != identity(entry, 'camera_frame') or state['step'] != 4000 or state['draws'][entry[1]['held_ids']].any():
            raise ValueError('Only the source-excluded final registered checkpoint may predict')
        torch.manual_seed(17); model = method.CameraFrameForecaster(reg['architecture'], entry[1]['baseline_index'], 'camera_frame')
        model.load_state_dict(state['model']); model.eval(); blocks, linear = [], []
        with torch.no_grad():
            for start in range(0, len(ids), 128):
                rows = ids[start:start+128]; q = data['camera_query_index'][rows]
                blocks.append(model(training.batch_inputs(data, rows, 'camera_frame')).numpy())
                linear.append(method.camera_cv(data['history'][rows], data['camera_past'][q], data['camera_prediction'][q])-data['origin'][rows, None])
                if start % 4096 == 0: beat(state='predicting_A_donors_only', job=key, rows=min(start+128, len(ids)))
        for name, value in (('camera_actor', np.concatenate(blocks)), ('camera_cv', np.concatenate(linear))):
            if not np.isfinite(value).all(): raise FloatingPointError('Nonfinite camera forecast')
            path = PRIVATE/'forecasts'/f'{key}_{name}.npz'; receipt = PUBLIC/'forecasts'/f'{key}_{name}.json'
            if receipt.exists():
                with np.load(checked(read(receipt)['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], data['global_ids'][ids]); np.testing.assert_array_equal(z['prediction'], value)
            else:
                if replay: raise ValueError('Missing original frozen prediction bank')
                path.parent.mkdir(parents=True, exist_ok=True); pending = path.with_suffix('.pending')
                with pending.open('wb') as f: np.savez_compressed(f, ids=data['global_ids'][ids], prediction=value)
                pending.replace(path); once(receipt, dict(rows=len(ids), forecast=artifact(path), kind=name, job=key))
            refs.append(artifact(receipt))
    if len(refs) != 12: raise ValueError('Six actor and six camera-CV banks required')
    if not replay: once(PUBLIC/'forecast_freeze.json', dict(banks=refs, all_predictions_frozen_before_scoring=True))


def evaluate(reg, data, jobs, beat, replay=False):
    if not read(PUBLIC/'forecast_freeze.json')['all_predictions_frozen_before_scoring']:
        raise ValueError('Freeze predictions before scoring')
    rows, recordings, strata, baselines = [], [], [], []
    for i, entry in enumerate(jobs):
        key, ids = entry[0]['key'], entry[2]; paths = {}
        with np.load(checked(reg['controls'][i]['forecast']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['global_ids'], data['global_ids'][ids]); paths['identity'] = z['relative_prediction'].astype(float)
        for name in ('camera_actor', 'camera_cv'):
            ref = read(PUBLIC/'forecasts'/f'{key}_{name}.json')
            with np.load(checked(ref['forecast']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], data['global_ids'][ids]); paths[name] = z['prediction'].astype(float)
        floor = metrics.baseline_numpy(data['history'][ids], 1)-data['origin'][ids, None]; paths['cv'] = floor
        target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
        cuts = reg['cuts'][str(entry[0]['producer_fold'])]
        active = data['camera_active'][data['camera_query_index'][ids]]
        for contrast, left, right in (('primary', 'identity', 'camera_actor'),
                ('camera_cv_effect', 'cv', 'camera_cv'), ('neural_over_camera_cv', 'camera_cv', 'camera_actor')):
            for groups, destination, name in ((data['sites'][ids], rows, 'source'), (data['recordings'][ids], recordings, 'recording')):
                for group in sorted(set(groups)):
                    at = groups == group
                    _, inverse, count = np.unique(data['queries'][ids][at], return_inverse=True, return_counts=True)
                    weights = 1./count[inverse]; weights /= weights.sum()
                    result = metrics.statistics(paths[left][at], paths[right][at], floor[at], target[at], valid[at], weights, cuts['easy'], cuts['hard'])
                    destination.append(dict(job=key, producer_fold=entry[0]['producer_fold'], contrast=contrast,
                        **{name: str(group)}, summary=result))
            for camera_ok in (False, True):
                at = active == camera_ok
                if at.any():
                    _, inverse, count = np.unique(data['queries'][ids][at], return_inverse=True, return_counts=True)
                    weights = 1./count[inverse]; weights /= weights.sum()
                    strata.append(dict(job=key, contrast=contrast, camera_active=camera_ok,
                        summary=metrics.statistics(paths[left][at], paths[right][at], floor[at], target[at], valid[at], weights, cuts['easy'], cuts['hard'])))
        for name, path in [('camera_compensated_cv', paths['camera_cv'])]+[
                (name, metrics.baseline_numpy(data['history'][ids], j)-data['origin'][ids, None]) for j, name in enumerate(BASELINES)]:
            for site in sorted(set(data['sites'][ids])):
                at = data['sites'][ids] == site
                _, inverse, count = np.unique(data['queries'][ids][at], return_inverse=True, return_counts=True)
                weights = 1./count[inverse]; weights /= weights.sum()
                baselines.append(dict(job=key, source=str(site), baseline=name,
                    summary=metrics.statistics(floor[at], path[at], floor[at], target[at], valid[at], weights, cuts['easy'], cuts['hard'])))
        beat(state='source_A_diagnostic_scored', job=key)
    primary = [r for r in rows if r['contrast'] == 'primary']
    if len(primary) != 14 or sum(r['summary'][-1]['rows'] for r in primary) != 125540:
        raise ValueError('Complete original source population required')
    lower = np.array([r['summary'][-1]['paired_completion_lower_pp'] for r in primary])
    known = np.array([r['summary'][-1]['paired_known_pp'] for r in primary])
    draws = np.random.default_rng(39107).integers(0, 14, (3000, 14))
    interval = np.quantile(lower[draws].mean(1), [.025, .975])
    folds = {str(f): float(np.mean([r['summary'][-1]['paired_known_pp'] for r in primary if r['producer_fold'] == f])) for f in (0, 1)}
    screen = bool(lower.mean() > 0 and interval[0] > 0 and (known > 0).sum() >= 8 and min(folds.values()) > 0)
    result = stable(dict(result_source='fresh_run', rows=125540, sources=rows, recordings=recordings, camera_strata=strata,
        causal_baselines=baselines, new_baselines_are_diagnostic_not_outcome_selected_deployment=True,
        primary=dict(known_pp=float(known.mean()), completion_lower_pp=float(lower.mean()), interval95=interval.tolist(),
            positive_known_sources=int((known > 0).sum()), fold_known_means=folds, component_progression_screen_pass=screen),
        one_forecaster_seed=True, independent_confirmation=False, original_rules_unchanged=True,
        new_controller_training=False, new_own_B_readout=False, new_outer_C_H_readout=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if replay: exact(read(PUBLIC/'analysis.json'), result)
    else: once(PUBLIC/'analysis.json', result)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('register', 'prepare', 'pilot', 'control', 'train', 'verify', 'forecast', 'evaluate', 'replay'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, reg = register()
        if args.phase == 'register': print('Registered fixed six-actor camera-frame test;no real fitting claimed.'); return
        committed(reg); start, guard, beat = runtime(cfg, args.phase)
        before = {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        if args.phase == 'prepare': prepare(cfg, beat)
        else:
            data, jobs = data_and_jobs()
            if args.phase == 'pilot': pilot(reg, data, jobs, guard, beat)
            elif args.phase == 'control': control(reg, data, jobs, guard, beat, args.resume)
            elif args.phase in ('train', 'verify'): train(reg, data, jobs, guard, beat, args.resume, args.phase == 'verify')
            elif args.phase == 'forecast': forecasts(reg, data, jobs, beat)
            elif args.phase == 'evaluate': evaluate(reg, data, jobs, beat)
            else:
                forecasts(reg, data, jobs, beat, replay=True); evaluate(reg, data, jobs, beat, replay=True)
                exact(before, {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()})
        destination = PRIVATE/'replays' if args.phase == 'replay' else PUBLIC/'executions'
        once(destination/f'{args.phase}_{time.time_ns()}_{os.getpid()}.json', dict(pid=os.getpid(), phase=args.phase,
            seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, actual_Torch=True))
        beat(state='phase_complete')


if __name__ == '__main__': main()
