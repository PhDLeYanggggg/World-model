"""Full matched shared-motion cost study on a separately verified forecast bank."""
import argparse
import fcntl
import gc
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
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.intake_m3w_movingdronecrowd import sha, once, atomic
from src.data_unification.m3w_moving_forecast_data import prepare, array_hash
from src.data_unification.m3w_moving_cost_bank import job_designs, bank
from src.world_model import m3w_shared_motion_cost as api
from src.world_model.m3w_european_source_forecast import fit_design
from src.world_model.m3w_frame_action_allocation import allocate, feasible, query_groups
from src.evaluation import m3w_selection_forecast_readout as readout
from src.evaluation.m3w_matched_intervention_utility import matched_utility

NAME = 'moving_shared_motion_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
PARENT = ROOT/'outputs/publication_readiness_2026_09/moving_source_forecast_supported_v1'
CONFIG = ROOT/'configs/m3w_moving_shared_motion_cost_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
    ROOT/'src/data_unification/m3w_moving_cost_bank.py',
    ROOT/'src/world_model/m3w_shared_motion_cost.py',
    ROOT/'tests/test_m3w_moving_cost_bank.py', ROOT/'tests/test_m3w_shared_motion_cost.py',
    ROOT/'src/world_model/m3w_bound_aware_cost.py', ROOT/'src/world_model/m3w_selection_controller.py',
    ROOT/'src/world_model/m3w_supported_bound_targets.py', ROOT/'src/world_model/m3w_floor_relative.py',
    ROOT/'src/world_model/m3w_multiaction_cost.py', ROOT/'src/world_model/m3w_reference_cost_refit.py',
    ROOT/'src/world_model/m3w_frame_action_allocation.py',
    ROOT/'src/world_model/m3w_european_source_intervention.py',
    ROOT/'src/world_model/m3w_geometric_cost_head.py', ROOT/'src/world_model/m3w_causal_descriptor_head.py',
    ROOT/'src/evaluation/m3w_selection_forecast_readout.py',
    ROOT/'tests/test_m3w_moving_shared_motion_runner.py',
    ROOT/'src/evaluation/m3w_matched_intervention_utility.py',
    ROOT/'tests/test_m3w_matched_intervention_utility.py')
MATCHED = ('initial_model', 'parameters', 'budget', 'sampler', 'draw_hash',
           'fit_ids', 'preprocess', 'torch_rng')


def read(path):
    return json.loads(path.read_text())


def stable(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def checked(ref):
    path = ROOT/ref['path']
    if sha(path) != ref['sha256']:
        raise ValueError('Changed frozen source artifact')
    return path


def runtime(cfg, phase):
    torch.set_num_threads(cfg['torch_threads'])
    torch.set_num_interop_threads(cfg['interop_threads'])
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged checkpoint')))
    start = time.monotonic()

    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Keep existing10GiB reserve, preserve complete work')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h phase boundary, resume original budget')

    def beat(**values):
        guard()
        total = sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
        if total > cfg['private_artifact_cap_bytes']:
            raise OSError('Registered private artifact cap exceeded')
        event = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            phase=phase, elapsed_seconds=time.monotonic()-start, private_artifact_bytes=total, **values)
        atomic(PRIVATE/'heartbeat.json', event)
        with (PRIVATE/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)

    beat(state='runtime_ready', machine=platform.machine(), device='cpu', threads=4, workers=0,
         resource_probing=False)
    return start, guard, beat


def admission(cfg, beat):
    for relative, expected in read(PARENT/'registration.json')['bindings'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError('Frozen forecast input/dependency code changed')
    training, verification, analysis = (read(PARENT/p) for p in
        ('training.json', 'verify_execution.json', 'analysis.json'))
    if (training['complete_primary_models'] != 18 or not training['matched_sampling_all9pairs_exact']
            or not verification['real_Torch_training'] or analysis['all_declared_group_primary_passed']
            or analysis['primary']['status'] != 'not_estimable'):
        raise ValueError('Complete verified original18-producer bank required')
    if (cfg['forecast_family_fixed_before_readout'] != 'eqmotion' or cfg['arms'] != list(api.ARMS)
            or cfg['new_models'] != 36 or cfg['input_dimension'] != api.DIMENSION
            or cfg['positive_harm_budget'] != .02 or cfg['easy_net_degradation_budget'] != .02
            or cfg['training']['query_batch_size'] != 72
            or any(cfg[k] for k in ('new_forecast_training', 'threshold_search', 'calibration_opened',
                'confirmation_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Keep frozen family, information contrast, full budget and original risk')
    source_cfg = read(ROOT/'configs/m3w_moving_source_forecast_supported_v1.json')
    data, population, source_admission = prepare(ROOT, source_cfg['admission'], beat)
    if population != read(PARENT/'population.json'):
        raise ValueError('Do not alter the original all-row source population')
    folds = source_admission['proposed_within_fit_threefold_groups']
    producers = []
    for row in read(PARENT/'job_roster.json')['jobs']:
        identity = row['identity']
        if identity['family'] != 'eqmotion':
            continue
        present = sorted(set(identity['fit_groups']) & set(data['sites']))
        design = fit_design(data, present, data['baseline_ade'])
        if design['baseline_index'] != identity['baseline_index']:
            raise ValueError('Recomputed producer fitting floor differs')
        producers.append(dict(identity, easy_cut=design['easy_cut'], hard_cut=design['hard_cut']))
    jobs = job_designs(data, folds, producers, cfg['training'])
    refs = {}
    for item in analysis['prediction_receipts']:
        checked(item)
        if item['identity']['family'] == 'eqmotion':
            identity = item['identity']
            key = f"eqmotion_fold{identity['fold']}_seed{identity['seed']}"
            refs[key] = item
    if len(refs) != 9:
        raise ValueError('All nine frozen EqMotion proposal arrays required')
    dependencies = [PARENT/p for p in ('registration.json', 'training.json', 'population.json',
        'analysis.json', 'job_roster.json', 'full_refit_verification.json', 'verify_execution.json')]
    value = stable(dict(configuration=cfg, bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent={p.name: artifact(p) for p in dependencies},
        jobs=jobs, forecasts=refs, actual_seeds=[17, 29, 43], input_dimension=api.DIMENSION,
        parameters=sum(p.numel() for p in api.base.base.CostHead(api.DIMENSION, 64, 17,
            np.zeros(5, np.float32), 'context').parameters()),
        numpy=np.__version__, torch=str(torch.__version__), machine=platform.machine(),
        source_rows=population['rows'], declared_groups=24, supported_groups=population['groups'],
        original_full_group_primary='not_estimable_not_passed',
        every_held_group_excluded_from_producer_and_critic=True,
        calibration_opened=False, confirmation_opened=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    once(REG, value)
    return data, jobs, refs


def verify_committed():
    for path in (*FILES, REG):
        content = subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
            cwd=ROOT, check=True, capture_output=True).stdout
        if hashlib.sha256(content).hexdigest() != sha(path):
            raise ValueError('Commit the complete definition before critic fitting')


def view(data, job, refs, training=False):
    with np.load(checked(refs[job['forecast_key']]), allow_pickle=False) as cache:
        return bank(data, job, cache['ids'], cache['relative_prediction'], training=training)


def path_for(job, arm, kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(data, job, arm, cfg, refs, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(data['learning'], arm=arm, seed=job['seed'], settings=cfg['training'],
        identity=dict(registration_sha256=sha(REG), job=job, forecast=refs[job['forecast_key']]),
        path=path_for(job, arm, kind), resume=resume, stop_at=stop_at, guard=guard,
        heartbeat=lambda **v: beat(job=job['key'], feature_arm=arm, **v))


def recovered(a, b):
    api.base.base.exact({k:v for k,v in a.items() if k != 'seconds'},
                        {k:v for k,v in b.items() if k != 'seconds'})


def pilot(cfg, data, jobs, refs, beat, guard, resume):
    cells = []
    for fold in cfg['pilot_critic_groups']:
        job = next(j for j in jobs if j['critic_fold'] == fold)
        fitting = view(data, job, refs, training=True)
        states = []
        for arm in cfg['arms']:
            path = path_for(job, arm, 'pilot_resume')
            if path.exists() and not resume:
                raise ValueError('Explicit resume required for partial pilot')
            step = api.base.base.read_checkpoint(path)['step'] if path.exists() else 0
            if step < cfg['pilot_stop']:
                one(fitting, job, arm, cfg, refs, beat, guard, kind='pilot_resume',
                    resume=path.exists(), stop_at=cfg['pilot_stop'])
            a = one(fitting, job, arm, cfg, refs, beat, guard, kind='pilot_resume',
                resume=True, stop_at=cfg['pilot_target'])
            direct_path = path_for(job, arm, 'pilot_direct')
            b = one(fitting, job, arm, cfg, refs, beat, guard, kind='pilot_direct',
                resume=resume and direct_path.exists(), stop_at=cfg['pilot_target'])
            recovered(a, b)
            states.append(a)
            cells.append(dict(job=job['key'], critic_fold=fold, arm=arm, step=a['step'],
                budget=a['budget'], fit_seconds=a['seconds'], parameters=a['parameters'],
                exact_resume_direct=True, checkpoint_bytes=path.stat().st_size))
        for key in MATCHED:
            api.base.base.exact(states[0][key], states[1][key])
        del fitting, states, a, b
        gc.collect()
    once(PUBLIC/'pilot.json', dict(cells=cells, engineering_pass=True,
        primary36fit_projection_seconds=6*sum(r['fit_seconds']/r['step']*r['budget'] for r in cells),
        projection_not_actual_full_completion=True, held_metrics_read=False))


def train(cfg, data, jobs, refs, beat, guard, resume):
    if not read(PUBLIC/'pilot.json')['engineering_pass']:
        raise ValueError('Complete real three-fold paired pilot first')
    fits = []
    for job in jobs:
        fitting = view(data, job, refs, training=True)
        states = []
        for arm in cfg['arms']:
            path = path_for(job, arm)
            state = one(fitting, job, arm, cfg, refs, beat, guard,
                resume=resume and path.exists())
            if state['step'] != job['fit_scope']['budget'] or state['sampler']['counts'].min() < 4:
                raise ValueError('Every complete query coverage budget must finish')
            if job == jobs[0]:
                refit_path = path_for(job, arm, 'full_refit')
                refit = one(fitting, job, arm, cfg, refs, beat, guard, kind='full_refit',
                    resume=resume and refit_path.exists())
                recovered(state, refit)
                once(PUBLIC/('first_full_refit_'+arm+'.json'),
                    dict(scientific_state_exact=True, step=state['step'], checkpoint=artifact(refit_path)))
                del refit
            pred = api.predict(state, fitting['causal'])
            receipt = dict(job=job, arm=arm, step=state['step'], checkpoint=artifact(path),
                parameters=state['parameters'], seconds=state['seconds'],
                minimum_query_visits=int(state['sampler']['counts'].min()),
                never_drawn_queries=int((state['sampler']['counts'] == 0).sum()),
                draw_hash=state['draw_hash'], fitting_predictions_sha256=array_hash(pred),
                initial_trace=state['trace'][0], final_trace=state['trace'][-1])
            output = PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            once(output, stable(receipt))
            fits.append(artifact(output))
            states.append(state)
            beat(state='critic_complete', job=job['key'], arm=arm, complete=len(fits), total=36)
            del pred
        for key in MATCHED:
            api.base.base.exact(states[0][key], states[1][key])
        del fitting, states, state
        gc.collect()
    once(PUBLIC/'training_freeze.json', dict(fits=fits, complete_heads=36,
        optimizer_updates=sum(read(checked(r))['step'] for r in fits), all18pairs_exact=True,
        real_Torch_training=True, calibration_opened=False, confirmation_opened=False))


def allocation_file(job, arm):
    return PRIVATE/'allocation'/(job['key']+'_'+arm+'.npz')


def allocated(pred, causal, held, job, arm, cfg, beat, *, verify=False):
    path = allocation_file(job, arm)
    saved = None
    if path.exists():
        with np.load(path, allow_pickle=False) as item:
            meta = json.loads(str(item['metadata_json']))
            saved = dict(meta, choices={key:item[key].copy() for key in
                ('independent', 'frame_uniform', 'joint_matched', 'joint_free')})
    if verify:
        if saved is None or saved['completed_queries'] != len(causal['groups']):
            raise ValueError('Complete original allocation receipts before verification')
        bindings = {k:hashlib.sha256(str((np.asarray(v).shape, np.asarray(v).dtype.str)).encode()
            +np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in zip(
            ('prediction', 'allowed', 'recordings', 'frames', 'ids'),
            (pred, causal['allowed'], held['recordings'], held['frames'], held['ids']))}
        bindings.update(seconds=cfg['joint_solver_seconds_per_query'], version=1)
        if saved['bindings'] != bindings:
            raise ValueError('Allocation was made from a different freshly replayed causal prediction')
        for ix in causal['groups']:
            for choice in saved['choices'].values():
                if not feasible(pred[ix], causal['allowed'][ix], choice[ix]):
                    raise ValueError('Stored allocation fails original predicted risk budget')
            if ((saved['choices']['joint_matched'][ix] >= 0).sum()
                    != (saved['choices']['independent'][ix] >= 0).sum()):
                raise ValueError('Stored utility control changed per-query intervention count')
        return saved['choices'], dict(cached_feasibility_verified=True,
            solver_outcome_receipt_retained_not_claimed_bitwise_reoptimization=True)

    def save(value):
        path.parent.mkdir(exist_ok=True)
        meta = {k:v for k,v in value.items() if k != 'choices'}
        temp = path.with_suffix('.pending.npz')
        np.savez_compressed(temp, metadata_json=np.asarray(json.dumps(meta, sort_keys=True)), **value['choices'])
        temp.replace(path)

    choice, _, report = allocate(pred, causal['allowed'], held['recordings'], held['frames'], held['ids'],
        seconds=cfg['joint_solver_seconds_per_query'], checkpoint=save, resume_state=saved,
        heartbeat=lambda **v: beat(state='joint_allocation', job=job['key'], arm=arm, **v))
    return choice, report


def policy_paths(paths, floor, choices):
    out = dict(floor=floor, uncontrolled_neural=paths[:, 6])
    for key, choice in choices.items():
        actual = floor.copy()
        ix = np.flatnonzero(choice >= 0)
        actual[ix] = paths[ix, choice[ix]]
        out[key] = actual
    return out


def group_interval(groups, getter, sites, cfg):
    value = readout.interval(groups, getter, expected_sites=sites,
        resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
    if 'by_locality' in value:
        value['by_source_group'] = value.pop('by_locality')
    value['bootstrap_unit'] = 'conservative_source_group_after_producer_and_seed_average_not_verified_IID_sites'
    value['scientific_role'] = 'supported_group_development_secondary_not_independent_confirmation'
    return value


def evaluate(cfg, data, jobs, refs, beat, guard, *, verify=False):
    freeze = read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads'] != 36 or not freeze['all18pairs_exact']:
        raise ValueError('Finish all36matched heads before held readout')
    models = {}
    for item in freeze['fits']:
        receipt = read(checked(item))
        models[receipt['job']['key'], receipt['arm']] = receipt
    groups, replays = [], []
    for job in jobs:
        held = view(data, job, refs)
        paths, ids, causal = held['paths'], held['ids'], held['causal']
        floor = paths[:, job['floor_index']]
        predictions, policies, allocations, allocated_choices = {}, {}, {}, {}
        for arm in cfg['arms']:
            receipt = models[job['key'], arm]
            state = api.base.base.read_checkpoint(checked(receipt['checkpoint']))
            prediction = api.predict(state, causal)
            predictions[arm] = array_hash(prediction)
            choice, allocations[arm] = allocated(prediction, causal, held, job, arm, cfg, beat, verify=verify)
            allocated_choices[arm] = choice
            policies[arm] = policy_paths(paths, floor, choice)
            del state, prediction
        rows = []
        for site in sorted(set(held['sites'])):
            at = held['sites'] == site
            target = data['target_eval'][ids[at]]-data['origin'][ids[at], None]
            valid = data['valid'][ids[at]]
            values = {}
            for arm in cfg['arms']:
                values[arm] = {key:readout.policy(value[at], floor[at], paths[at, 1], target, valid,
                    held['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut'])
                    for key,value in policies[arm].items()}
            paired = {key:readout.paired(policies['local_history'][key][at],
                policies['shared_motion'][key][at], floor[at], target, valid) for key in cfg['policies']}
            local_groups = query_groups(held['recordings'][at], held['frames'][at], ids[at])
            matched = {arm:matched_utility(policies[arm]['independent'][at],
                policies[arm]['joint_matched'][at], floor[at], target, valid,
                allocated_choices[arm]['independent'][at], allocated_choices[arm]['joint_matched'][at],
                local_groups) for arm in cfg['arms']}
            rows.append(dict(site=str(site), job=job['key'], seed=job['seed'],
                producer_fold=job['producer_fold'], critic_fold=job['critic_fold'],
                policies=values, paired_shared_minus_local=paired,
                within_arm_nonempty_matched_utility=matched))
        output = PUBLIC/'groups'/(job['key']+'.json')
        value = stable(dict(job=job, rows=rows, prediction_hashes=predictions))
        if verify:
            if not output.exists() or read(output) != value:
                raise ValueError('Fresh36-head cost inference and every held policy scalar must replay exactly')
        else:
            once(output, value)
            once(PUBLIC/'allocation_reports'/(job['key']+'.json'), stable(allocations))
        groups.extend(rows)
        replays.append(artifact(output))
        beat(state='held_job_verified' if verify else 'held_job_complete', job=job['key'], rows=len(rows))
        del held, paths, policies, causal, allocated_choices
        gc.collect()
    sites = sorted(set(data['sites']))
    expected = {(site, f, s) for site in sites for f in range(3) for s in cfg['seeds']
        if f != next(j['evaluation_fold'] for j in jobs if site in j['evaluation_sites'])}
    observed = {(r['site'], r['producer_fold'], r['seed']) for r in groups}
    if len(groups) != 6*len(sites) or observed != expected:
        raise ValueError('Every supported group needs both excluded producers and three actual seeds')
    interval = lambda getter: group_interval(groups, getter, sites, cfg)
    contrasts = {key:{metric:interval(lambda r,k=key,m=metric:r['paired_shared_minus_local'][k][m])
        for metric in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for key in cfg['policies']}
    matched_contrasts = {arm:{metric:interval(lambda r,a=arm,m=metric:
        r['within_arm_nonempty_matched_utility'][a]['contrast'][m])
        for metric in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for arm in cfg['arms']}
    matched_support = {arm:dict(nonempty_query_views=sum(r['within_arm_nonempty_matched_utility'][arm]['nonempty_queries'] for r in groups),
        empty_query_views=sum(r['within_arm_nonempty_matched_utility'][arm]['empty_queries'] for r in groups),
        nonestimable_source_views=sum(r['within_arm_nonempty_matched_utility'][arm]['contrast']['known_pp'] is None for r in groups))
        for arm in cfg['arms']}
    safety = {}
    accuracy, lower, subset_gains = {}, {}, {}
    for arm in cfg['arms']:
        safety[arm] = {}
        accuracy[arm] = {}
        lower[arm] = {}
        subset_gains[arm] = {}
        for key in cfg['policies']:
            accuracy[arm][key] = interval(lambda r,a=arm,k=key:
                r['policies'][a][k]['known_ADE_gain_floor_percent'])
            lower[arm][key] = interval(lambda r,a=arm,k=key:
                r['policies'][a][k]['completion_lower_gain_floor_percent'])
            subset_gains[arm][key] = {subset:interval(lambda r,a=arm,k=key,s=subset:
                r['policies'][a][k]['subsets'][s]['gain_percent']) for subset in ('easy', 'hard')}
            records = [v for r in groups for v in r['policies'][arm][key]['recordings'].values()]
            values = [r['known_easy_net_degradation'] for r in records if r['known_easy_net_degradation'] is not None]
            safety[arm][key] = dict(required_record_views=len(records),
                supported_record_views=sum(r['supported'] for r in records),
                full_original_risk_pass=bool(records and all(r['supported'] for r in records)),
                undefined_easy_views=sum(r['completion_easy_risk_upper'] is None for r in records),
                known_easy_risk_violations=sum(r['known_easy_risk'] is not None and r['known_easy_risk']>.02 for r in records),
                known_all_risk_violations=sum(r['known_all_risk'] is not None and r['known_all_risk']>.02 for r in records),
                worst_known_easy_net_degradation_percent=100*max(values) if values else None)
    summary = stable(dict(scope='source_development_not_independent_confirmation',
        original24group_primary='not_estimable_not_passed', supported_groups_secondary=len(sites),
        paired_shared_minus_local=contrasts, safety=safety, group_receipts=replays,
        known_accuracy=accuracy, completion_lower=lower, subset_gains=subset_gains,
        within_arm_nonempty_matched_utility=matched_contrasts, matched_utility_support=matched_support,
        matched_utility_is_not_safety_gate=True, count_matching_within_arm_not_between_feature_arms=True,
        feature_effect_uses_full_original_population=True,
        declared_feature_secondary=contrasts['joint_free']['completion_lower_pp'],
        actual_training_seeds=cfg['seeds'], unique_past_eligible_rows=len(data['sites']),
        independent_physical_sites_not_established=True, independent_confirmation=False,
        new_deployment=False, stage5c_executed=False, smc_enabled=False))
    once(PUBLIC/'summary.json', summary)
    if verify:
        for arm in cfg['arms']:
            one_refit = read(PUBLIC/('first_full_refit_'+arm+'.json'))
            original = api.base.base.read_checkpoint(checked(models[jobs[0]['key'], arm]['checkpoint']))
            replay = api.base.base.read_checkpoint(checked(one_refit['checkpoint']))
            recovered(original, replay)
        once(PUBLIC/'verification.json', dict(all36cost_predictions_and_policy_scores_exact=True,
            full_first_refits_exact=True, summary_sha256=sha(PUBLIC/'summary.json'),
            solver_receipts_and_original2percent_feasibility_checked=True,
            new_solver_optimality_or_bitwise_reoptimization_not_claimed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'prepare', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        end = PUBLIC/(args.phase+'_execution.json')
        if end.exists():
            raise ValueError('Complete phase immutable')
        cfg = read(CONFIG)
        start, guard, beat = runtime(cfg, args.phase)
        data, jobs, refs = admission(cfg, beat)
        if args.phase == 'register':
            return
        verify_committed()
        if args.phase == 'pilot':
            pilot(cfg, data, jobs, refs, beat, guard, args.resume)
        elif args.phase == 'train':
            train(cfg, data, jobs, refs, beat, guard, args.resume)
        elif args.phase in ('evaluate', 'verify'):
            evaluate(cfg, data, jobs, refs, beat, guard, verify=args.phase == 'verify')
        once(end, dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            result_source='fresh_run' if args.phase != 'verify' else 'fresh_saved_head_replay_and_cached_solver_feasibility',
            real_Torch_training=args.phase in ('pilot', 'train'),
            source_only=True, calibration_opened=False, confirmation_opened=False,
            stage5c_executed=False, smc_enabled=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
