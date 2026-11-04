"""Matched neural cost objective repair; no new forecast or official data roles."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_hurdle_reference as lineage
from src.world_model import m3w_record_excess_cost as api
from src.evaluation.m3w_verified_choice_cache import verified_choices, POLICIES
from src.evaluation.m3w_selected_cost_audit import selected_costs

source = lineage.source
PARENT = lineage.lineage.parent
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/moving_record_excess_cost_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/moving_record_excess_cost_v1'
CONFIG = ROOT / 'configs/m3w_moving_record_excess_cost_v1.json'
REG = PUBLIC / 'registration.json'
RUNTIME_REG = PUBLIC / 'runtime_registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC / 'protocol.md', ROOT / 'src/world_model/m3w_record_excess_cost.py',
         ROOT / 'tests/test_m3w_record_excess_cost.py')


def register(beat):
    spec = source.read(CONFIG)
    if (spec['training'] != source.read(PARENT.CONFIG)['training'] or spec['loss_weight'] != 1
            or spec['positive_harm_budget'] != .02 or spec['easy_net_degradation_budget'] != .02
            or spec['new_primary_neural_cost_heads'] != 18 or spec['primary_optimizer_updates'] != 36000
            or any(spec[k] for k in ('threshold_search', 'new_data_roles', 'new_forecast_training',
                'confirmation_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Keep the matched objective-only18fit comparison')
    _, cfg, data, jobs, refs = lineage.register(beat)
    verification = source.read(ROOT / 'outputs/publication_readiness_2026_09/moving_selected_reference_v1/verification.json')
    if not verification['all4176selected_reference_rows_and_metrics_exact']:
        raise ValueError('Complete the fixed selected-region diagnosis first')
    controls = {source.read(source.checked(r))['job']['key']: source.read(source.checked(r))
        for r in source.read(PARENT.PUBLIC / 'training_freeze.json')['fits']
        if source.read(source.checked(r))['arm'] == 'amplitude_bank'}
    if len(controls) != 18:
        raise ValueError('All18actual matched neural controls required')
    value = dict(configuration=spec,
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        parent_registration=source.artifact(PARENT.REG), controls={k: r['checkpoint'] for k, r in controls.items()},
        selected_diagnostic=source.artifact(ROOT / 'outputs/publication_readiness_2026_09/moving_selected_reference_v1/analysis.json'),
        jobs=jobs, forecasts=refs, new_neural_cost_training=True, new_trajectory_training=False,
        new_data_roles=False, teacher_C_masks_are_supervised_training_not_OOF_validation=True,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False)
    original = source.read(REG)
    for key in ('configuration', 'controls', 'jobs', 'forecasts'):
        api.base.base.exact(source.stable(value[key]), source.stable(original[key]))
    source.once(RUNTIME_REG, source.stable(dict(value, scientific_registration=source.artifact(REG),
        runtime_patch=source.artifact(PUBLIC / 'runtime_patch_note.md'),
        revision='logging_duplicate_total_fixed_objective_data_budget_unchanged')))
    return spec, cfg, data, jobs, refs, controls


def state_path(job, arm, kind='fits'):
    return PRIVATE / kind / (job['key'] + '_' + arm + '.pt.gz')


def fitting(data, job, refs, control):
    p = PARENT.view(data, job, refs, training=True)
    state = api.base.base.read_checkpoint(source.checked(control['checkpoint']))
    frozen = api.parent.predict(state, p['causal'])
    if source.array_hash(frozen) != control['fitting_predictions_sha256']:
        raise ValueError('Fitting teacher must match the actual original neural control')
    teacher, _ = lineage.lineage.api.previous.select_actions(frozen, p['causal']['allowed'], arm='physical_neural')
    p['learning'] = api.training_packet(p['learning'], teacher, p['recordings'])
    return p, state, frozen


def identity(job, arm):
    return dict(registration_sha256=source.sha(REG), job=job, record_loss_arm=arm,
        feature_arm='local_history', amplitude_arm='amplitude_bank',
        amplitude_schema='floor_then_six_fixed_fractions_true_CV_separate')


def one(p, job, arm, spec, beat, guard, *, kind='fits', resume=False, stop_at=None):
    return api.fit(p['learning'], loss_arm=arm, seed=job['seed'], settings=spec['training'],
        identity=identity(job, arm), path=state_path(job, arm, kind), resume=resume, stop_at=stop_at,
        heartbeat=lambda **v: beat(job=job['key'], **v), guard=guard)


def pilot(spec, cfg, data, jobs, refs, controls, beat, guard, resume):
    cells = []
    for fold in range(3):
        job = next(j for j in jobs if j['critic_fold'] == fold)
        p, _, _ = fitting(data, job, refs, controls[job['key']])
        for arm in api.ARMS:
            path = state_path(job, arm, 'pilot_resume')
            step = api.base.base.read_checkpoint(path)['step'] if path.exists() else 0
            if path.exists() and not resume:
                raise ValueError('Explicit unchanged pilot resume required')
            if step < spec['pilot_stop']:
                one(p, job, arm, spec, beat, guard, kind='pilot_resume', resume=path.exists(), stop_at=spec['pilot_stop'])
            recovered = one(p, job, arm, spec, beat, guard, kind='pilot_resume', resume=True, stop_at=spec['pilot_target'])
            direct = one(p, job, arm, spec, beat, guard, kind='pilot_direct',
                resume=resume and state_path(job, arm, 'pilot_direct').exists(), stop_at=spec['pilot_target'])
            api.exact_scientific_state(recovered, direct)
            if arm == 'point':
                original_path = state_path(job, arm, 'pilot_original_loop')
                original = api.parent.fit(p['learning'], arm='amplitude_bank', seed=job['seed'], settings=spec['training'],
                    identity=identity(job, arm), path=original_path, heartbeat=lambda **v: beat(job=job['key'], **v),
                    guard=guard, resume=resume and original_path.exists(), stop_at=spec['pilot_target'])
                api.exact_scientific_state(direct, original)
            cells.append(dict(job=job['key'], arm=arm, seconds=direct['seconds'], steps=direct['step'],
                budget=direct['budget'], parameters=direct['parameters'], exact_resume=True,
                original_loop_exact=True if arm == 'point' else None))
        del p, recovered, direct
        source.gc.collect()
        guard()
    projection = 6 * sum(r['seconds']/r['steps']*r['budget'] for r in cells if r['arm'] == 'record_excess')
    source.once(PUBLIC / 'pilot.json', source.stable(dict(cells=cells, all6resume_cells_exact=True,
        all3original_point_loops_exact=True, actual_pilot_optimizer_updates=1500,
        projected_primary_fit_seconds_not_completion=projection)))


def train(spec, cfg, data, jobs, refs, controls, beat, guard, resume):
    if not source.read(PUBLIC / 'pilot.json')['all3original_point_loops_exact']:
        raise ValueError('Complete real matched-loop and resume pilot first')
    receipts = []
    for job in jobs:
        p, old, frozen = fitting(data, job, refs, controls[job['key']])
        path = state_path(job, 'record_excess')
        state = one(p, job, 'record_excess', spec, beat, guard, resume=resume and path.exists())
        for key in ('initial_model', 'parameters', 'budget', 'sampler', 'draw_hash', 'fit_ids', 'preprocess', 'settings', 'seed', 'torch_rng'):
            api.base.base.exact(state[key], old[key])
        if state['step'] != 2000:
            raise ValueError('Complete the unchanged final2000update budget')
        if job == jobs[0]:
            point_path = state_path(job, 'point', 'full_refit')
            point = one(p, job, 'point', spec, beat, guard, kind='full_refit', resume=resume and point_path.exists())
            api.exact_scientific_state(point, old)
            replica_path = state_path(job, 'record_excess', 'full_refit')
            replica = one(p, job, 'record_excess', spec, beat, guard, kind='full_refit', resume=resume and replica_path.exists())
            api.exact_scientific_state(replica, state)
            source.once(PUBLIC / 'first_full_refits.json', dict(point_matches_original=True,
                record_excess_refit_exact=True, point_checkpoint=source.artifact(point_path),
                record_checkpoint=source.artifact(replica_path)))
            del point, replica
        predicted = api.parent.predict(state, p['causal'])
        truth = api.parent.targets(p, data['target_eval'][p['ids']]-data['origin'][p['ids'], None],
                                   data['valid'][p['ids']], job['easy_cut'])
        teacher = p['learning']['teacher_actions']
        path_receipt = PUBLIC / 'fits' / (job['key'] + '.json')
        source.once(path_receipt, source.stable(dict(job=job, checkpoint=source.artifact(path),
            cached_matched_control=controls[job['key']]['checkpoint'], matched_initialization_sampling_normalization_budget=True,
            step=state['step'], seconds=state['seconds'], parameters=state['parameters'],
            trace=state['trace'], fitting_predictions_sha256=source.array_hash(predicted),
            fixed_teacher_region={name: selected_costs(value, truth['moments'], truth['observed_future'], teacher, p['causal']['groups'])
                for name, value in (('point', frozen), ('record_excess', predicted))},
            active_runtime_registration=source.artifact(RUNTIME_REG),
            new_neural_cost_training=True, new_trajectory_training=False)))
        receipts.append(source.artifact(path_receipt))
        beat(state='record_cost_fit_frozen', job=job['key'], complete=len(receipts), total=18)
        del p, old, frozen, state, predicted, truth
        source.gc.collect()
        guard()
    source.once(PUBLIC / 'training_freeze.json', dict(fits=receipts, complete_new_neural_cost_heads=18,
        primary_optimizer_updates=36000, cached_matched_controls=18,
        all_fits_before_new_Hprediction=True, new_trajectory_training=False, new_data_roles=False))


def evaluate(spec, cfg, data, jobs, refs, controls, beat, guard, verify):
    freeze = source.read(PUBLIC / 'training_freeze.json')
    if freeze['complete_new_neural_cost_heads'] != 18:
        raise ValueError('Freeze all18full-budget models before new held prediction')
    models = {source.read(source.checked(r))['job']['key']: source.read(source.checked(r)) for r in freeze['fits']}
    solver = {(r['job'], r['arm']): r for r in source.read(lineage.lineage.PUBLIC / 'solver_summary.json')}
    records, receipts = [], []
    for job in jobs:
        p = PARENT.view(data, job, refs, training=False)
        c, ids = p['causal'], p['ids']
        old = api.base.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        new = api.base.base.read_checkpoint(source.checked(models[job['key']]['checkpoint']))
        predicted = dict(point=api.parent.predict(old, c), record_excess=api.parent.predict(new, c))
        prior = source.read(PARENT.PUBLIC / 'groups' / (job['key'] + '.json'))
        if source.array_hash(predicted['point']) != prior['cost_hashes']['amplitude_bank']:
            raise ValueError('Fresh matched point cost output changed')
        choices = {}
        for name, base, tag in (('point', PARENT.PRIVATE, 'amplitude_bank'),
                               ('shared_tree', lineage.lineage.PRIVATE, 'shared_motion')):
            arm = 'frozen' if name == 'point' else 'shared_motion'
            choices[name], _ = verified_choices(base / 'allocation' / (job['key'] + '_' + tag + '.npz'),
                solver[job['key'], arm]['sha256'], c, p, cfg['joint_solver_seconds_per_query'])
        source.PRIVATE = PRIVATE
        choices['record_excess'], _ = source.allocated(predicted['record_excess'], c, p, job, 'record_excess', cfg, beat, verify=verify)
        hashes = {a: source.array_hash(v) for a, v in predicted.items()}
        chashes = {a: {k: source.array_hash(v) for k, v in choice.items()} for a, choice in choices.items()}
        paths = {a: source.policy_paths(p['paths'], p['floor'], choice) for a, choice in choices.items()}
        rows = []
        shared_old = source.read(lineage.lineage.PUBLIC / 'groups' / (job['key'] + '.json'))
        for site in sorted(set(p['sites'])):
            at = p['sites'] == site
            target, valid = data['target_eval'][ids[at]]-data['origin'][ids[at], None], data['valid'][ids[at]]
            policies = {a: {k: source.readout.policy(value[at], p['floor'][at], p['cv'][at], target, valid,
                p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for k, value in bank.items()}
                for a, bank in paths.items()}
            for arm, expected, tag in (('point', prior, 'amplitude_bank'), ('shared_tree', shared_old, 'shared_motion')):
                earlier = next(r for r in expected['rows'] if r['site'] == str(site))['policies'][tag]
                api.base.base.exact(source.stable(policies[arm]), source.stable(earlier))
            paired = {a: {k: source.readout.paired(paths[a][k][at], paths['record_excess'][k][at], p['floor'][at], target, valid)
                for k in POLICIES} for a in ('point', 'shared_tree')}
            q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
            matched = {a: source.matched_utility(paths[a]['independent'][at], paths[a]['joint_matched'][at],
                p['floor'][at], target, valid, choices[a]['independent'][at], choices[a]['joint_matched'][at], q) for a in paths}
            rows.append(dict(site=str(site), producer_fold=job['producer_fold'], seed=job['seed'], policies=policies,
                             paired=paired, matched_utility=matched))
        if hashes != {a: source.array_hash(v) for a, v in predicted.items()} or chashes != {
                a: {k: source.array_hash(v) for k, v in choice.items()} for a, choice in choices.items()}:
            raise ValueError('Evaluation labels changed predictions or choices')
        path = PUBLIC / 'held' / (job['key'] + '.json')
        source.once(path, source.stable(dict(job=job, rows=rows, prediction_hashes=hashes, choice_hashes=chashes,
            cached_point_and_shared_policy_scalars_exact=True, old_neural_costs_fresh=True, shared_tree_costs_fresh=False)))
        receipts.append(source.artifact(path))
        records.extend(rows)
        beat(state='record_cost_held_replayed' if verify else 'record_cost_held_evaluated', job=job['key'], complete=len(receipts), total=18)
        del p, c, old, new, predicted, paths, choices, rows
        source.gc.collect()
        guard()
    if len(records) != 138:
        raise ValueError('Retain all138supported-source views')
    sites = sorted({r['site'] for r in records})
    ci = lambda get: source.group_interval(records, get, sites, cfg)
    paired = {a: {p: {m: ci(lambda r, a=a, p=p, m=m: r['paired'][a][p][m]) for m in
        ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for p in POLICIES} for a in ('point', 'shared_tree')}
    safety, accuracy, lower = {}, {}, {}
    for arm in ('point', 'shared_tree', 'record_excess'):
        safety[arm], accuracy[arm], lower[arm] = {}, {}, {}
        for policy in POLICIES:
            views = [v for r in records for v in r['policies'][arm][policy]['recordings'].values()]
            easy = [v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
            safety[arm][policy] = dict(required_record_views=len(views), supported_record_views=sum(v['supported'] for v in views),
                known_all_risk_violations=sum(v['known_all_risk'] is not None and v['known_all_risk'] > .02 for v in views),
                known_easy_risk_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk'] > .02 for v in views),
                undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
                worst_known_easy_net_degradation_percent=100*max(easy) if easy else None,
                full_original_risk_pass=bool(views and all(v['supported'] for v in views)))
            accuracy[arm][policy] = ci(lambda r, arm=arm, policy=policy: r['policies'][arm][policy]['known_ADE_gain_floor_percent'])
            lower[arm][policy] = ci(lambda r, arm=arm, policy=policy: r['policies'][arm][policy]['completion_lower_gain_floor_percent'])
    source.once(PUBLIC / 'summary.json', source.stable(dict(paired=paired, safety=safety, known_accuracy=accuracy,
        completion_lower=lower, group_receipts=receipts, new_neural_cost_heads=18, new_trajectory_training=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    if verify:
        source.once(PUBLIC / 'verification.json', dict(all36neural_cost_outputs_and_all_policy_scalar_blocks_exact=True,
            old_shared_costs_freshly_inferred=False, bitwise_MILP_reoptimization_claimed=False,
            summary_sha256=source.sha(PUBLIC / 'summary.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train', 'evaluate', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    source.PRIVATE = PRIVATE
    lineage.PRIVATE = PRIVATE
    with (PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        spec = source.read(CONFIG)
        cfg = source.read(PARENT.CONFIG)
        start, guard, beat = lineage.runtime(cfg, spec, args.phase)
        spec, cfg, data, jobs, refs, controls = register(beat)
        if args.phase == 'register':
            return
        for path in (*FILES, RUNTIME_REG):
            content = source.subprocess.run(['git', 'show', 'HEAD:' + str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if source.hashlib.sha256(content).hexdigest() != source.sha(path):
                raise ValueError('Commit unchanged objective definition before fitting')
        receipt = PUBLIC / (args.phase + '_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase is immutable')
        if args.phase == 'pilot':
            pilot(spec, cfg, data, jobs, refs, controls, beat, guard, args.resume)
        elif args.phase == 'train':
            train(spec, cfg, data, jobs, refs, controls, beat, guard, args.resume)
        else:
            evaluate(spec, cfg, data, jobs, refs, controls, beat, guard, args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_neural_cost_training=args.phase in ('pilot', 'train'), new_trajectory_training=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
