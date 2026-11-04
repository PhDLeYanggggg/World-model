"""Fixed adaptive/frozen/point/tree comparison after both neural banks freeze."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_adaptive_record_cost as run
from src.evaluation.m3w_verified_choice_cache import verified_choices, POLICIES, binding
from src.evaluation.m3w_selected_cost_audit import selected_costs

source, PUBLIC, PRIVATE = run.source, run.PUBLIC, run.PRIVATE
REG = PUBLIC / 'evaluation_registration.json'
FILES = (Path(__file__), PUBLIC / 'evaluation_protocol.md')
RUNTIME_REG = PUBLIC / 'evaluation_runtime_registration.json'
PATCH_NOTE = PUBLIC / 'evaluation_runtime_patch_note.md'
RUNTIME_FILES = (*FILES, ROOT / 'tests/test_m3w_adaptive_record_evaluation.py', PATCH_NOTE)
ARMS = ('point', 'shared_tree', 'frozen_teacher', 'adaptive')


def check_choice_hashes(choices, expected):
    if set(choices) != set(POLICIES) or {k: source.array_hash(v) for k, v in choices.items()} != expected:
        raise ValueError('Every cached control choice must match its verified parent readout')


def check_prediction_binding(meta, prediction):
    if meta['original_prediction_binding'] != binding(prediction):
        raise ValueError('Cached neural choice bank must bind the actual current cost output')


def runtime_registration():
    original = source.read(REG)
    if ((PUBLIC / 'evaluate_execution.json').exists() or any((PUBLIC / 'held').glob('*.json'))
            or any((PRIVATE / 'allocation').glob('*.npz'))):
        raise ValueError('Record this serialization repair before adaptive choices or scoring')
    if original['bindings'][str(FILES[1].relative_to(ROOT))] != source.sha(FILES[1]):
        raise ValueError('Do not change the original scientific protocol')
    for name in ('adaptive_training_registration', 'frozen_training_registration', 'C_only_trigger'):
        source.checked(original[name])
    source.once(RUNTIME_REG, source.stable(dict(
        **{k: v for k, v in original.items() if k != 'bindings'},
        original_evaluation_registration=source.artifact(REG),
        bindings={str(p.relative_to(ROOT)): source.sha(p) for p in RUNTIME_FILES},
        failed_preallocation_log=source.artifact(PRIVATE / 'evaluate_failed_binding_20261104.log'),
        failed_pid=74361, failed_exit_code_observed=1,
        repair='use_original_solver_shape_dtype_serialization_for_its_prediction_binding',
        frozen_models_source_roles_protocol_losses_thresholds_and_metrics_unchanged=True,
        new_adaptive_labels_not_scored_before_patch=True)))


def registration():
    if RUNTIME_REG.exists():
        current, original = source.read(RUNTIME_REG), source.read(REG)
        source.checked(current['original_evaluation_registration'])
        if any(current.get(k) != v for k, v in original.items() if k != 'bindings'):
            raise ValueError('Runtime repair must preserve every scientific registration field')
        if any(source.sha(ROOT / filename) != expected for filename, expected in current['bindings'].items()):
            raise ValueError('Corrected runtime definition changed')
        return RUNTIME_REG
    if not REG.exists() and ((run.teacher.PUBLIC / 'evaluate_execution.json').exists()
            or any((run.teacher.PUBLIC / 'held').glob('*.json'))):
        raise ValueError('Register the full comparison before any new Houtcome')
    source.once(REG, source.stable(dict(bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        adaptive_training_registration=source.artifact(run.REG),
        frozen_training_registration=source.artifact(run.teacher.RUNTIME_REG),
        C_only_trigger=source.artifact(run.teacher.PUBLIC / 'fitting_regions/analysis.json'),
        arms=list(ARMS), primary='adaptive_minus_frozen_teacher_joint_free_completion_lower',
        strong_control='adaptive_minus_shared_tree_joint_free_completion_lower',
        no_H_model_or_threshold_selection=True, independent_confirmation=False)))
    return REG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'runtime-register', 'evaluate', 'verify'))
    args = parser.parse_args()
    if args.phase == 'runtime-register':
        runtime_registration()
        return
    active_registration = registration()
    if args.phase == 'register':
        return
    definition = source.read(active_registration)
    for path in (*(ROOT / filename for filename in definition['bindings']), active_registration):
        content = source.subprocess.run(['git', 'show', 'HEAD:' + str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
        if source.hashlib.sha256(content).hexdigest() != source.sha(path):
            raise ValueError('Commit unchanged full-comparison definition before readout')
    if (PUBLIC / (args.phase + '_execution.json')).exists():
        raise ValueError('Completed evaluation is immutable')
    if args.phase == 'verify' and not (PUBLIC / 'summary.json').exists():
        raise ValueError('Complete adaptive evaluation before replay')
    freeze = source.read(PUBLIC / 'training_freeze.json')
    if freeze['complete_new_adaptive_cost_heads'] != 18:
        raise ValueError('All18adaptive models must freeze before any new Hcomparison')
    parent_v = source.read(run.teacher.PUBLIC / 'verification.json')
    if (not parent_v['all36neural_cost_outputs_and_all_policy_scalar_blocks_exact']
            or parent_v['summary_sha256'] != source.sha(run.teacher.PUBLIC / 'summary.json')):
        raise ValueError('Complete and replay the frozen-objective evaluation first')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    source.PRIVATE = PRIVATE
    run.teacher.lineage.PRIVATE = PRIVATE
    with (PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        spec, cfg = source.read(run.CONFIG), source.read(run.teacher.PARENT.CONFIG)
        start, guard, beat = run.teacher.lineage.runtime(cfg, spec, args.phase)
        spec, cfg, data, jobs, refs, controls, frozen = run.register(beat)
        models = {source.read(source.checked(r))['job']['key']: source.read(source.checked(r)) for r in freeze['fits']}
        parent_rows = {source.read(source.checked(r))['job']['key']: source.read(source.checked(r))
            for r in source.read(run.teacher.PUBLIC / 'summary.json')['group_receipts']}
        rows, receipts = [], []
        for job in jobs:
            p = run.teacher.PARENT.view(data, job, refs, training=False)
            c, ids = p['causal'], p['ids']
            paths_ref = dict(point=controls[job['key']]['checkpoint'],
                frozen_teacher=frozen[job['key']]['checkpoint'], adaptive=models[job['key']]['checkpoint'])
            predictions = {name: run.api.parent.predict(run.api.base.base.read_checkpoint(source.checked(ref)), c)
                for name, ref in paths_ref.items()}
            original = parent_rows[job['key']]
            for name, old_name in (('point', 'point'), ('frozen_teacher', 'record_excess')):
                if source.array_hash(predictions[name]) != original['prediction_hashes'][old_name]:
                    raise ValueError('Fresh matched neural controls must match parent cost hashes')
            choices = {}
            locations = dict(point=(run.teacher.PARENT.PRIVATE, 'amplitude_bank'),
                shared_tree=(run.teacher.lineage.lineage.PRIVATE, 'shared_motion'),
                frozen_teacher=(run.teacher.PRIVATE, 'record_excess'))
            for name, (base, tag) in locations.items():
                path = base / 'allocation' / (job['key'] + '_' + tag + '.npz')
                choices[name], meta = verified_choices(path, source.sha(path), c, p, cfg['joint_solver_seconds_per_query'])
                old_name = 'record_excess' if name == 'frozen_teacher' else name
                check_choice_hashes(choices[name], original['choice_hashes'][old_name])
                if name != 'shared_tree':
                    check_prediction_binding(meta, predictions[name])
            source.PRIVATE = PRIVATE
            choices['adaptive'], _ = source.allocated(predictions['adaptive'], c, p, job, 'adaptive_selection', cfg, beat,
                                                       verify=args.phase == 'verify')
            hashes = {name: source.array_hash(v) for name, v in predictions.items()}
            chashes = {name: {k: source.array_hash(v) for k, v in choice.items()} for name, choice in choices.items()}
            paths = {name: source.policy_paths(p['paths'], p['floor'], choice) for name, choice in choices.items()}
            groups = []
            for site in sorted(set(p['sites'])):
                at = p['sites'] == site
                target, valid = data['target_eval'][ids[at]]-data['origin'][ids[at], None], data['valid'][ids[at]]
                policies = {name: {k: source.readout.policy(v[at], p['floor'][at], p['cv'][at], target, valid,
                    p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for k, v in bank.items()}
                    for name, bank in paths.items()}
                previous = next(r for r in original['rows'] if r['site'] == str(site))
                for name, old_name in (('point', 'point'), ('shared_tree', 'shared_tree'), ('frozen_teacher', 'record_excess')):
                    run.api.base.base.exact(source.stable(policies[name]), source.stable(previous['policies'][old_name]))
                paired = {name: {k: source.readout.paired(paths[name][k][at], paths['adaptive'][k][at], p['floor'][at], target, valid)
                    for k in POLICIES} for name in ARMS[:-1]}
                q = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
                matched = {name: source.matched_utility(paths[name]['independent'][at], paths[name]['joint_matched'][at],
                    p['floor'][at], target, valid, choices[name]['independent'][at], choices[name]['joint_matched'][at], q) for name in ARMS}
                ce, _ = source.readout.errors(p['cv'][at], target, valid)
                zero = valid.any(1) & (ce == 0)
                zero_cv = {name: dict(known_zero_CV_rows=int(zero.sum()),
                    positive_error_on_zero_CV_rows=int((zero & (source.readout.errors(paths[name]['joint_free'][at], target, valid)[0] > 0)).sum()))
                    for name in ARMS}
                groups.append(dict(site=str(site), producer_fold=job['producer_fold'], seed=job['seed'],
                    policies=policies, paired=paired, matched_utility=matched, zero_CV_diagnostic=zero_cv))
            truth = run.api.parent.targets(p, data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids], job['easy_cut'])
            cost_diagnostic = [dict(model=model, region=region, scores=selected_costs(predictions[model], truth['moments'],
                truth['observed_future'], choices[region]['joint_free'], c['groups']))
                for model in ('frozen_teacher', 'adaptive') for region in ('frozen_teacher', 'adaptive', 'shared_tree')]
            if hashes != {name: source.array_hash(v) for name, v in predictions.items()} or chashes != {
                    name: {k: source.array_hash(v) for k, v in choice.items()} for name, choice in choices.items()}:
                raise ValueError('Future labels changed predictions or cached choices')
            path = PUBLIC / 'held' / (job['key'] + '.json')
            source.once(path, source.stable(dict(job=job, rows=groups, prediction_hashes=hashes, choice_hashes=chashes,
                fixed_region_cost_diagnostics=cost_diagnostic, labels_used_only_after_all_choices=True,
                cached_parent_policy_scalar_blocks_exact=True, shared_tree_costs_fresh=False)))
            receipts.append(source.artifact(path))
            rows.extend(groups)
            beat(state='adaptive_held_replayed' if args.phase == 'verify' else 'adaptive_held_evaluated', job=job['key'],
                 complete=len(receipts), total=18)
            del p, c, predictions, choices, paths, groups, truth, cost_diagnostic
            source.gc.collect()
            guard()
        if len(rows) != 138:
            raise ValueError('All138supported-source views required')
        sites = sorted({r['site'] for r in rows})
        ci = lambda fn: source.group_interval(rows, fn, sites, cfg)
        paired = {a: {p: {m: ci(lambda r, a=a, p=p, m=m: r['paired'][a][p][m]) for m in
            ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for p in POLICIES} for a in ARMS[:-1]}
        safety, known, lower = {}, {}, {}
        for arm in ARMS:
            safety[arm], known[arm], lower[arm] = {}, {}, {}
            for policy in POLICIES:
                views = [v for r in rows for v in r['policies'][arm][policy]['recordings'].values()]
                easy = [v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
                safety[arm][policy] = dict(required_record_views=len(views), supported_record_views=sum(v['supported'] for v in views),
                    known_all_risk_violations=sum(v['known_all_risk'] is not None and v['known_all_risk'] > .02 for v in views),
                    known_easy_risk_violations=sum(v['known_easy_risk'] is not None and v['known_easy_risk'] > .02 for v in views),
                    undefined_easy_views=sum(v['completion_easy_risk_upper'] is None for v in views),
                    worst_known_easy_net_degradation_percent=100*max(easy) if easy else None,
                    full_original_risk_pass=bool(views and all(v['supported'] for v in views)))
                known[arm][policy] = ci(lambda r, arm=arm, policy=policy: r['policies'][arm][policy]['known_ADE_gain_floor_percent'])
                lower[arm][policy] = ci(lambda r, arm=arm, policy=policy: r['policies'][arm][policy]['completion_lower_gain_floor_percent'])
        source.once(PUBLIC / 'summary.json', source.stable(dict(paired=paired, safety=safety, known_accuracy=known,
            completion_lower=lower, group_receipts=receipts, fresh_neural_cost_arrays=54,
            shared_tree_costs_freshly_inferred=False, original24group_primary='not_estimable_not_passed',
            independent_confirmation=False, new_deployment=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))
        source.once(PUBLIC / (args.phase + '_execution.json'), dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss, new_training=False))
        if args.phase == 'verify':
            source.once(PUBLIC / 'verification.json', dict(all54neural_cost_arrays_and_policy_blocks_exact=True,
                all_three_cached_parent_policy_blocks_exact=True, bitwise_MILP_reoptimization_claimed=False,
                summary_sha256=source.sha(PUBLIC / 'summary.json')))
        beat(state='complete')


if __name__ == '__main__':
    main()
