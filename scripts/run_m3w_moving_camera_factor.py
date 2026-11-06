"""Full matched source-held camera-operator ablation on already exposed A data."""
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
from scripts import run_m3w_moving_camera_frame as parent
from scripts.run_m3w_moving_source_forecast import exact, same_scientific_checkpoint
from src.world_model import m3w_camera_factor as method

source, metrics, training = parent.source, parent.metrics, parent.training
np, torch = parent.np, parent.torch
read, sha, stable, once, checked, artifact = parent.read, parent.sha, parent.stable, parent.once, parent.checked, parent.artifact
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_camera_factor_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_camera_factor_v1'
CONFIG = ROOT/'configs/m3w_moving_camera_factor_v1.json'; REG = PUBLIC/'registration.json'
ARMS = ('past_only', 'future_only')
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', Path(method.__file__), ROOT/'tests/test_m3w_camera_factor.py')


def register():
    cfg = read(CONFIG); _, old = parent.register(); outcome = read(parent.PUBLIC/'analysis.json')
    if (cfg['treatment_arms'] != list(ARMS) or cfg['fresh_treatment_models'] != 12
            or cfg['forecaster_seeds'] != [17] or cfg['steps_per_model'] != old['training']['steps']
            or cfg['steps_per_model'] != 4000 or cfg['original_risk_budget'] != .02
            or cfg['primary'] != 'past_only_minus_identity_whole12_paired_completion_lower'
            or cfg['uninterrupted_optimizer_budget'] != 60200
            or outcome['primary']['component_progression_screen_pass']
            or any(cfg[k] for k in ('warm_start', 'threshold_search', 'identity_certification',
                'new_own_B_readout', 'new_outer_C_H_readout', 'deployment_changed', 'submission_ready',
                'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed full-budget operator ablation and original evaluation rules required')
    if not read(parent.PUBLIC/'verification.json')['full_refits_exact']:
        raise ValueError('Completed parent evidence required')
    controls = []
    for old_control in old['controls']:
        key = old_control['job']['key']; both_fit = read(parent.PUBLIC/'fits'/f'{key}.json')
        record = dict(job=key, identity=dict(checkpoint=old_control['checkpoint'], forecast=old_control['forecast'],
            ids_key='global_ids', prediction_key='relative_prediction'),
            both=dict(checkpoint=both_fit['checkpoint'], forecast=read(parent.PUBLIC/'forecasts'/f'{key}_camera_actor.json')['forecast'],
                ids_key='ids', prediction_key='prediction'),
            camera_cv=dict(forecast=read(parent.PUBLIC/'forecasts'/f'{key}_camera_cv.json')['forecast'],
                ids_key='ids', prediction_key='prediction'))
        for value in (record['identity'], record['both'], record['camera_cv']):
            checked(value['forecast'])
            if 'checkpoint' in value: checked(value['checkpoint'])
        controls.append(record)
    if len(controls) != 6: raise ValueError('Six complete cached source designs required')
    dependency_paths = [Path(parent.__file__), Path(training.__file__),
        ROOT/'src/world_model/m3w_camera_frame.py', ROOT/'src/world_model/m3w_context_conditioning.py',
        ROOT/'src/world_model/m3w_native_forecast.py', ROOT/'src/world_model/m3w_european_source_forecast.py',
        ROOT/'src/world_model/m3w_eqmotion_adapter.py', Path(source.__file__), Path(metrics.__file__)]
    reg = stable(dict(config=cfg, architecture=old['architecture'], training=old['training'], jobs=old['jobs'],
        author_core=old['author_core'], cuts=old['cuts'], controls=controls,
        files={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in dependency_paths},
        parents=[artifact(parent.PUBLIC/name) for name in ('registration_v2.json', 'population.json',
            'training.json', 'verification.json', 'analysis.json', 'forecast_freeze.json')],
        torch=str(torch.__version__), numpy=np.__version__,
        original_rules_unchanged=True, independent_confirmation=False, new_own_B_readout=False,
        new_outer_C_H_readout=False, role='exposed_producer_A_component_not_independent_test'))
    once(REG, reg)
    once(PUBLIC/'input_admission.json', dict(registration_sha256=sha(REG),
        role='supervised_training_previously_exposed_producer_A_only',
        observed_inputs=['past_agent_neighbor_geometry', 'past_image_affine_proxy'],
        future_operator_is_past_only_extrapolation=True, future_targets_or_masks_in_model_input=False,
        current_risk_support_NA_rules_unchanged=True, identity_certification=False,
        new_private_geometry_or_image_copy=False))
    return cfg, reg


def committed(reg):
    for p in (*FILES, REG, PUBLIC/'input_admission.json'):
        value = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT,
            capture_output=True, check=True).stdout
        if hashlib.sha256(value).hexdigest() != sha(p): raise ValueError('Commit exact definitions before real execution')
    for path, digest in reg['dependencies'].items():
        if sha(ROOT/path) != digest: raise ValueError('Changed pinned parent implementation')


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged checkpoint')))
    start, last = time.monotonic(), [0.]
    def guard():
        now = time.monotonic()
        if now-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h phase boundary;resume unchanged budget')
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
        with (PRIVATE/'events.jsonl').open('a') as out: out.write(source.json.dumps(event)+'\n')
        print(source.json.dumps(event), flush=True)
    beat(state='ready', machine=platform.machine(), device='cpu', threads=4, interop=1, workers=0, resource_probing=False)
    return start, guard, beat


def data_and_jobs(reg):
    data, jobs = parent.data_and_jobs()
    if len(jobs) != 6 or len(data['sites']) != 125540: raise ValueError('Complete original population required')
    for entry, control in zip(jobs, reg['controls']):
        if entry[0]['key'] != control['job']: raise ValueError('Changed source-job order')
        for arm in ('identity', 'both', 'camera_cv'):
            spec = control[arm]
            with np.load(checked(spec['forecast']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z[spec['ids_key']], data['global_ids'][entry[2]])
    return data, jobs


def identity(entry, arm):
    return dict(parent_identity=entry[3], operator_arm=arm, registration_sha256=sha(REG))


def fit_one(data, entry, reg, home, arm, guard, beat, resume=False, stop_at=None):
    torch.manual_seed(17); model = method.CameraFactorForecaster(reg['architecture'], entry[1]['baseline_index'], arm)
    result = training.fit(model, data, entry[1], arm='identity' if arm == 'identity' else 'camera_frame',
        seed=17, settings=reg['training'], identity=identity(entry, arm), directory=home, resume=resume,
        stop_at=stop_at, guard=guard, heartbeat=lambda **kw: beat(job=entry[0]['key'], arm=arm, **kw))
    state = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
    if (result['parameters'] != 3027268 or result['trainable_parameters'] != 580676
            or state['draws'][entry[1]['held_ids']].any()):
        raise ValueError('Changed parameter capacity or held-source exposure')
    return result, state


def pilot(reg, data, jobs, guard, beat):
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot is immutable')
    a, b = PRIVATE/'pilot/resumed', PRIVATE/'pilot/direct'
    if not (a/'checkpoint.pt').exists(): fit_one(data, jobs[0], reg, a, 'past_only', guard, beat, stop_at=100)
    result, left = fit_one(data, jobs[0], reg, a, 'past_only', guard, beat, resume=True, stop_at=200)
    direct, right = fit_one(data, jobs[0], reg, b, 'past_only', guard, beat,
        resume=(b/'checkpoint.pt').exists(), stop_at=200)
    same_scientific_checkpoint(left, right)
    once(PUBLIC/'pilot.json', stable(dict(result_source='fresh_run', actual_Torch=True,
        real100plus100_equals200=True, full_scientific_state_exact=True, prefix=artifact(a/'checkpoint.pt'),
        direct200_seconds=direct['seconds'], projected15_full_fits_seconds=direct['seconds']/200*60000,
        nominal_parameters=result['parameters'], trainable_parameters=result['trainable_parameters'],
        not_predictive_success=True)))


def control(reg, data, jobs, guard, beat, resume):
    expected = identity(jobs[0], 'identity')
    if parent.complete_receipt(PUBLIC/'control.json', expected) is not None: return
    result, new = fit_one(data, jobs[0], reg, PRIVATE/'control', 'identity', guard, beat, resume)
    old = torch.load(checked(reg['controls'][0]['identity']['checkpoint']), map_location='cpu', weights_only=False)
    for key in ('model', 'optimizer', 'draws', 'train_ids', 'factors', 'sampler_rng', 'torch_rng', 'seed', 'step', 'settings'):
        exact(old[key], new[key])
    x, y = ({r['step']: r for r in value['losses']} for value in (old, new))
    if set(y)-set(x) or set(x)-set(y)-{20, 40}: raise ValueError('Changed original common loss trace')
    for step in y: exact(x[step], y[step])
    once(PUBLIC/'control.json', dict(identity=expected, result=result,
        checkpoint=artifact(PRIVATE/'control/checkpoint.pt'), complete4000shared_state_and_common_trace_exact=True))


def train(reg, data, jobs, guard, beat, resume, verify=False):
    if not read(PUBLIC/'control.json')['complete4000shared_state_and_common_trace_exact']:
        raise ValueError('Full original-control parity required first')
    receipts = []; prefix = read(PUBLIC/'pilot.json')['prefix']
    for arm in ARMS:
        for i, entry in enumerate(jobs[:1] if verify else jobs):
            key = entry[0]['key']; home = PRIVATE/('verify' if verify else 'fits')/arm/key
            receipt_path = PUBLIC/('refits' if verify else 'fits')/arm/f'{key}.json'
            done = parent.complete_receipt(receipt_path, identity(entry, arm))
            if done is not None:
                receipts.append(done['checkpoint']); beat(state='completed_fit_verified', arm=arm, complete=len(receipts)); continue
            if not verify and arm == 'past_only' and i == 0 and not (home/'checkpoint.pt').exists():
                home.mkdir(parents=True, exist_ok=True); shutil.copyfile(checked(prefix), home/'checkpoint.pt')
            result, state = fit_one(data, entry, reg, home, arm, guard, beat, resume or (home/'checkpoint.pt').exists())
            if not result['complete']: raise ValueError('Full4000update budget required')
            if verify:
                original = torch.load(checked(read(PUBLIC/'fits'/arm/f'{key}.json')['checkpoint']), map_location='cpu', weights_only=False)
                same_scientific_checkpoint(original, state)
            else:
                old = torch.load(checked(reg['controls'][i]['identity']['checkpoint']), map_location='cpu', weights_only=False)
                for k in ('draws', 'factors', 'train_ids', 'sampler_rng', 'torch_rng'): exact(old[k], state[k])
            receipt = dict(result_source='fresh_run', job=entry[0], arm=arm, result=result,
                identity=identity(entry, arm), checkpoint=artifact(home/'checkpoint.pt'),
                original_draws_factors_and_RNG_exact=True, full_refit_exact=verify)
            once(receipt_path, stable(receipt)); receipts.append(receipt['checkpoint'])
            beat(state='full_refit_exact' if verify else 'fit_complete', arm=arm, complete=len(receipts))
    once(PUBLIC/('verification.json' if verify else 'training.json'), dict(models=len(receipts),
        steps_each=4000, full_refits_exact=verify, first_design_per_arm_only=verify, checkpoints=receipts))


def forecasts(reg, data, jobs, beat, replay=False):
    verification = read(PUBLIC/'verification.json')
    if verification['models'] != 2 or not verification['full_refits_exact']: raise ValueError('Both full treatment refits required')
    refs = []
    for arm in ARMS:
        for entry in jobs:
            key = entry[0]['key']; ids = entry[2]; fit = read(PUBLIC/'fits'/arm/f'{key}.json')
            state = torch.load(checked(fit['checkpoint']), map_location='cpu', weights_only=False)
            if state['identity'] != identity(entry, arm) or state['step'] != 4000 or state['draws'][entry[1]['held_ids']].any():
                raise ValueError('Only source-excluded final checkpoint may predict')
            torch.manual_seed(17); model = method.CameraFactorForecaster(reg['architecture'], entry[1]['baseline_index'], arm)
            model.load_state_dict(state['model']); model.eval(); blocks = []
            with torch.no_grad():
                for start in range(0, len(ids), 128):
                    rows = ids[start:start+128]
                    blocks.append(model(training.batch_inputs(data, rows, 'camera_frame')).numpy())
                    if start % 4096 == 0: beat(state='predicting_A_donors_only', job=key, arm=arm, rows=min(start+128, len(ids)))
            value = np.concatenate(blocks)
            if not np.isfinite(value).all(): raise FloatingPointError('Nonfinite factorial forecast')
            path = PRIVATE/'forecasts'/arm/f'{key}.npz'; receipt = PUBLIC/'forecasts'/arm/f'{key}.json'
            if receipt.exists():
                with np.load(checked(read(receipt)['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], data['global_ids'][ids]); np.testing.assert_array_equal(z['prediction'], value)
            else:
                if replay: raise ValueError('Missing original frozen prediction bank')
                path.parent.mkdir(parents=True, exist_ok=True); pending = path.with_suffix('.pending')
                with pending.open('wb') as out: np.savez_compressed(out, ids=data['global_ids'][ids], prediction=value)
                pending.replace(path); once(receipt, dict(rows=len(ids), arm=arm, job=key, forecast=artifact(path)))
            refs.append(artifact(receipt))
    if len(refs) != 12: raise ValueError('Freeze the entire two-arm matrix')
    if not replay: once(PUBLIC/'forecast_freeze.json', dict(banks=refs, all_new_predictions_frozen_before_scoring=True))


def evaluate(reg, data, jobs, beat, replay=False):
    if not read(PUBLIC/'forecast_freeze.json')['all_new_predictions_frozen_before_scoring']:
        raise ValueError('Freeze predictions before scoring')
    sources, recordings, strata, baselines = [], [], [], []
    for entry, control in zip(jobs, reg['controls']):
        key, ids = entry[0]['key'], entry[2]; paths = {}
        for arm in ('identity', 'both', 'camera_cv'):
            spec = control[arm]
            with np.load(checked(spec['forecast']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z[spec['ids_key']], data['global_ids'][ids])
                paths[arm] = z[spec['prediction_key']].astype(float)
        for arm in ARMS:
            ref = read(PUBLIC/'forecasts'/arm/f'{key}.json')
            with np.load(checked(ref['forecast']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], data['global_ids'][ids]); paths[arm] = z['prediction'].astype(float)
        floor = metrics.baseline_numpy(data['history'][ids], 1)-data['origin'][ids, None]; paths['cv'] = floor
        target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
        cuts = reg['cuts'][str(entry[0]['producer_fold'])]; active = data['camera_active'][data['camera_query_index'][ids]]
        contrasts = (('primary', 'identity', 'past_only'), ('remove_future', 'both', 'past_only'),
            ('future_effect', 'identity', 'future_only'), ('add_past_to_future', 'future_only', 'both'),
            ('both_effect', 'identity', 'both'), ('past_over_camera_cv', 'camera_cv', 'past_only'),
            ('camera_cv_effect', 'cv', 'camera_cv'))
        def score(at, left, right):
            _, inverse, count = np.unique(data['queries'][ids][at], return_inverse=True, return_counts=True)
            weights = 1./count[inverse]; weights /= weights.sum()
            return metrics.statistics(left[at], right[at], floor[at], target[at], valid[at], weights, cuts['easy'], cuts['hard'])
        for contrast, left, right in contrasts:
            for groups, destination, label in ((data['sites'][ids], sources, 'source'), (data['recordings'][ids], recordings, 'recording')):
                for group in sorted(set(groups)):
                    destination.append(dict(job=key, producer_fold=entry[0]['producer_fold'], contrast=contrast,
                        **{label: str(group)}, summary=score(groups == group, paths[left], paths[right])))
            for camera_ok in (False, True):
                at = active == camera_ok
                if at.any(): strata.append(dict(job=key, contrast=contrast, camera_active=camera_ok,
                    summary=score(at, paths[left], paths[right])))
        for name, path in [('camera_compensated_cv', paths['camera_cv'])]+[
                (name, metrics.baseline_numpy(data['history'][ids], j)-data['origin'][ids, None]) for j, name in enumerate(parent.BASELINES)]:
            for site in sorted(set(data['sites'][ids])):
                baselines.append(dict(job=key, source=str(site), baseline=name,
                    summary=score(data['sites'][ids] == site, floor, path)))
        beat(state='source_A_diagnostic_scored', job=key)
    primary = [r for r in sources if r['contrast'] == 'primary']
    if len(primary) != 14 or sum(r['summary'][-1]['rows'] for r in primary) != 125540:
        raise ValueError('Complete original source population required')
    previous = read(parent.PUBLIC/'analysis.json')
    exact(previous['causal_baselines'], baselines)
    for now in (r for r in sources if r['contrast'] == 'both_effect'):
        old = next(r for r in previous['sources'] if r['contrast'] == 'primary' and r['source'] == now['source'])
        exact(old['summary'], now['summary'])
    lower = np.array([r['summary'][-1]['paired_completion_lower_pp'] for r in primary])
    known = np.array([r['summary'][-1]['paired_known_pp'] for r in primary])
    draws = np.random.default_rng(39107).integers(0, 14, (3000, 14))
    interval = np.quantile(lower[draws].mean(1), [.025, .975])
    folds = {str(f): float(np.mean([r['summary'][-1]['paired_known_pp'] for r in primary if r['producer_fold'] == f])) for f in (0, 1)}
    interaction = []
    for row in primary:
        selected = {c: next(r for r in sources if r['source'] == row['source'] and r['contrast'] == c)
            for c in ('primary', 'future_effect', 'both_effect')}
        interaction.append(dict(source=row['source'], known_pp_by_prefix=[
            selected['both_effect']['summary'][i]['paired_known_pp']-selected['primary']['summary'][i]['paired_known_pp']
            -selected['future_effect']['summary'][i]['paired_known_pp'] for i in range(4)]))
    result = stable(dict(result_source='fresh_diagnostic_readout_with_cached_verified_identity_both_controls',
        rows=125540, sources=sources, recordings=recordings, camera_strata=strata, causal_baselines=baselines,
        parent_both_and_baseline_statistics_exact=True, descriptive_known_interaction=interaction,
        interaction_missing_label_effect_identified=False,
        primary=dict(known_pp=float(known.mean()), completion_lower_pp=float(lower.mean()), interval95=interval.tolist(),
            positive_known_sources=int((known > 0).sum()), fold_known_means=folds,
            component_progression_screen_pass=bool(lower.mean() > 0 and interval[0] > 0 and (known > 0).sum() >= 8 and min(folds.values()) > 0)),
        one_forecaster_seed=True, independent_confirmation=False, original_rules_unchanged=True,
        new_controller_training=False, new_own_B_readout=False, new_outer_C_H_readout=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if replay: exact(read(PUBLIC/'analysis.json'), result)
    else: once(PUBLIC/'analysis.json', result)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('register', 'admit', 'pilot', 'control', 'train', 'verify', 'forecast', 'evaluate', 'replay'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, reg = register()
        if args.phase == 'register': print('Registered fixed operator ablation;no real execution claimed.'); return
        committed(reg); start, guard, beat = runtime(cfg, args.phase)
        before = {str(p): sha(p) for p in PUBLIC.rglob('*') if p.is_file()}
        data, jobs = data_and_jobs(reg)
        if args.phase == 'admit':
            once(PUBLIC/'population.json', dict(rows=len(data['sites']), sources=14, recordings=34, queries=1613,
                result_source='cached_verified_operator_geometry_and_forecast_row_alignment',
                cached_bank_arms=['identity', 'both', 'camera_cv'], all18cached_bank_row_alignments_exact=True,
                all_rows_retained=True, new_geometry_copy=False, generic_label_bank_opened_for_integrity=True,
                future_targets_or_masks_in_model_inputs=False, new_own_B_C_H_outcomes_opened=False))
        elif args.phase == 'pilot': pilot(reg, data, jobs, guard, beat)
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
