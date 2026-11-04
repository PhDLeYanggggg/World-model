"""Fresh reference inference, frozen choices and selected-region diagnostics."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_hurdle_reference as run
from src.evaluation.m3w_selected_reference_audit import selected_reference
from src.evaluation.m3w_verified_choice_cache import verified_choices, POLICIES

source, api = run.source, run.api
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/moving_selected_reference_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/moving_selected_reference_v1'
REG = PUBLIC / 'registration_v2.json'
FILES = (Path(__file__), PUBLIC / 'protocol.md',
    ROOT / 'src/evaluation/m3w_selected_reference_audit.py', ROOT / 'tests/test_m3w_selected_reference_audit.py')
ARMS = ('frozen', 'local_zero', 'shared_motion')
MODELS = ('old_neural', 'direct', 'hurdle', 'C_selected')


def registration():
    v = source.read(run.PUBLIC / 'strong_control/verification.json')
    if not v['all_arrays_and_two_estimand_metrics_exact']:
        raise ValueError('Complete the strong learned-control comparison first')
    source.once(REG, source.stable(dict(bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        supersedes_unexecuted_preflight=source.artifact(PUBLIC / 'registration.json'),
        revision_reason='Clarify exact-count versus floating mass tolerance before any diagnostic execution',
        head_freeze=source.artifact(run.PUBLIC / 'training_freeze.json'),
        strong_control=source.artifact(run.PUBLIC / 'strong_control/verification.json'),
        solver_ledger=source.artifact(run.lineage.PUBLIC / 'solver_summary.json'),
        original_cost_summary=source.artifact(run.lineage.PUBLIC / 'summary.json'),
        result_source='fresh_inference_then_cached_verified_fixed_choice_arithmetic',
        new_training=False, new_decisions=False, threshold_search=False,
        post_readout_diagnostic=True, independent_confirmation=False)))


def cache_predictions(job, p, fit, verify):
    path = PRIVATE / (job['key'] + '.npz')
    if verify:
        receipt = source.read(PUBLIC / 'groups' / (job['key'] + '.json'))
        source.checked(receipt['reference_cache'])
        with source.np.load(path, allow_pickle=False) as data:
            source.np.testing.assert_array_equal(data['ids'], p['ids'])
            predictions = {m: data[m].copy() for m in MODELS}
        return predictions
    states = {m['task']: api.base.read_checkpoint(source.checked(m['checkpoint']))
        for m in fit['models'] if m['partition'] == 'full'}
    pred = api.predictions(states, p['causal'])
    pred['prior'] = source.np.full(len(p['ids']), fit['full_prior']['reference'])
    pred['prior_event'] = source.np.full(len(p['ids']), fit['full_prior']['event'])
    expected = source.read(run.PUBLIC / 'held' / (job['key'] + '.json'))['prediction_hashes']
    if {k: source.array_hash(v) for k, v in pred.items()} != expected:
        raise ValueError('All reference predictions must match the original completed evaluation')
    ref = source.read(run.lineage.REG)['controls'][job['key']]
    old = run.lineage.api.previous.parent.predict(api.base.read_checkpoint(source.checked(ref)), p['causal'])
    original = source.read(run.lineage.PUBLIC / 'groups' / (job['key'] + '.json'))
    if source.array_hash(old) != original['cost_hashes']['frozen']:
        raise ValueError('Fresh old neural cost output must match the frozen hash')
    predictions = {m: pred[m] * p['causal']['scales'] for m in ('direct', 'hurdle')}
    predictions['C_selected'] = pred[fit['selected_by_C_OOF']] * p['causal']['scales']
    predictions['old_neural'] = old[:, 6, 3]
    if path.exists():
        with source.np.load(path, allow_pickle=False) as data:
            for m, v in dict(predictions, ids=p['ids']).items():
                source.np.testing.assert_array_equal(data[m], v)
    else:
        source.np.savez_compressed(path, **predictions, ids=p['ids'])
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    registration()
    if args.phase == 'register':
        return
    for path in (*FILES, REG):
        content = source.subprocess.run(['git', 'show', 'HEAD:' + str(path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, check=True).stdout
        if source.hashlib.sha256(content).hexdigest() != source.sha(path):
            raise ValueError('Commit the diagnostic definition before computing')
    if (PUBLIC / (args.phase + '_execution.json')).exists():
        raise ValueError('Completed phase is immutable')
    verify = args.phase == 'verify'
    if verify and not (PUBLIC / 'run_execution.json').exists():
        raise ValueError('Complete inference/arithmetic before cached replay')
    run.PRIVATE = PRIVATE
    source.PRIVATE = PRIVATE
    spec = source.read(run.CONFIG)
    cfg = source.read(run.lineage.parent.CONFIG)
    start, guard, beat = run.runtime(cfg, spec, 'selected_reference_' + args.phase)
    spec, cfg, data, jobs, refs = run.register(beat)
    fits = {source.read(source.checked(r))['job']['key']: source.read(source.checked(r))
        for r in source.read(run.PUBLIC / 'training_freeze.json')['fits']}
    solvers = {(r['job'], r['arm']): r for r in source.read(run.lineage.PUBLIC / 'solver_summary.json')}
    records, receipts = [], []
    for job in jobs:
        p = run.packet(data, job, refs, False)
        predictions = cache_predictions(job, p, fits[job['key']], verify)
        choices = {}
        for arm in ARMS:
            base = run.lineage.parent.PRIVATE if arm == 'frozen' else run.lineage.PRIVATE
            tag = 'amplitude_bank' if arm == 'frozen' else arm
            choices[arm], _ = verified_choices(base / 'allocation' / (job['key'] + '_' + tag + '.npz'),
                solvers[job['key'], arm]['sha256'], p['causal'], p, cfg['joint_solver_seconds_per_query'])
        hashes = {a: {k: source.array_hash(v) for k, v in ch.items()} for a, ch in choices.items()}
        y, known = run.labels(data, p, job)
        y = y * p['causal']['scales']
        original = source.read(run.lineage.PUBLIC / 'held_costs' / (job['key'] + '.json'))
        rows = []
        for arm in ARMS:
            for policy in POLICIES:
                for record in sorted(set(p['recordings'])):
                    at = p['recordings'] == record
                    result = selected_reference(y[at], known[at], p['causal']['scales'][at],
                        choices[arm][policy][at], {m: v[at] for m, v in predictions.items()})
                    old = original['diagnostics'][arm][policy]['recordings'][str(record)]
                    check = (result['selected'] == old['selected']
                        and result['selected_known'] == old['selected_known']
                        and source.np.isclose(result['actual_known_ER_mass'], old['realized_selected_known_moments'][3], rtol=1e-12, atol=1e-9)
                        and source.np.isclose(result['models']['old_neural']['predicted_known_ER_mass'],
                            old['predicted_on_selected_known_moments'][3], rtol=1e-12, atol=1e-9))
                    if not check:
                        raise ValueError('Original selected counts and reference masses must agree')
                    rows.append(dict(job=job['key'], site=str(p['sites'][at][0]), recording=str(record),
                        arm=arm, policy=policy, **result))
        if hashes != {a: {k: source.array_hash(v) for k, v in ch.items()} for a, ch in choices.items()}:
            raise ValueError('Supervision changed a frozen choice')
        path = PUBLIC / 'groups' / (job['key'] + '.json')
        source.once(path, source.stable(dict(job=job, rows=rows,
            reference_cache=source.artifact(PRIVATE / (job['key'] + '.npz')),
            choice_hashes=hashes, original_counts_exact_reference_masses_rtol1e_12_atol1e_9=True)))
        records.extend(rows)
        receipts.append(source.artifact(path))
        beat(state='selected_reference_complete', job=job['key'], complete=len(receipts), total=18)
        del p, predictions, choices, y, known, rows
        source.gc.collect()
        guard()
    if len(records) != 4176:
        raise ValueError('Retain all4176original arm-policy-record views')
    effects, counts = {}, {}
    for arm in ARMS:
        effects[arm], counts[arm] = {}, {}
        for policy in POLICIES:
            rows = [r for r in records if r['arm'] == arm and r['policy'] == policy]
            counts[arm][policy] = dict(views=len(rows), active=sum(r['selected'] > 0 for r in rows),
                active_without_observed_ER=sum(r['selected'] > 0 and not r['observed_reference_support'] for r in rows),
                active_without_ER_all_selected_have_some_label=sum(r['selected'] > 0 and not r['observed_reference_support']
                    and r['selected_unknown'] == 0 for r in rows),
                positive_prediction_without_observed_support={m: sum(r['models'][m]['positive_prediction_without_observed_support']
                    for r in rows) for m in MODELS})
            effects[arm][policy] = {loss: {m: source.group_interval(rows,
                lambda r, m=m, loss=loss: None if r['models'][m][loss] is None else
                    r['models']['old_neural'][loss] - r['models'][m][loss],
                sorted(set(data['sites'])), cfg) for m in MODELS[1:]}
                for loss in ('known_row_normalized_MSE', 'known_record_normalized_mass_squared_error')}
    source.once(PUBLIC / 'analysis.json', source.stable(dict(counts=counts, effects=effects, receipts=receipts,
        all4176_counts_exact_reference_masses_rtol1e_12_atol1e_9=True,
        future_labels_used_only_after_fixed_predictions_and_choices=True,
        post_readout_diagnostic=True, new_training=False, new_decisions=False, new_risk_gate=False,
        original24group_primary='not_estimable_not_passed', independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    source.once(PUBLIC / (args.phase + '_execution.json'), dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
        fresh_model_inference=not verify, choices_cached_verified=True, new_training=False))
    if verify:
        source.once(PUBLIC / 'verification.json', dict(all4176selected_reference_rows_and_metrics_exact=True,
            analysis_sha256=source.sha(PUBLIC / 'analysis.json'), fresh_model_inference=False, new_decisions=False))
    beat(state='complete')


if __name__ == '__main__':
    main()
