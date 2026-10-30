"""Read-only original-versus-final selected-score audit of frozen critics."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_query_unit_critic as unit
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_selected_score_audit import audit
from src.evaluation.m3w_metric_bundle import pack

parent = unit.parent; np = unit.np
NAME = 'european_selected_score_audit_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_selected_score_audit.py',
    ROOT/'tests/test_m3w_selected_score_audit.py', PUBLIC/'protocol.md')
sha, once, write = unit.sha, unit.once, unit.write


def registration(register=False):
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        original_registration_sha256=sha(parent.REG), original_summary_sha256=sha(parent.PUBLIC/'summary.json'),
        unit_registration_sha256=sha(unit.REG), unit_training_freeze_sha256=sha(unit.PUBLIC/'training_freeze.json'),
        groups=72, variants=8, changed_models_or_policies=False, new_optimizer_updates=0,
        risk_budget=.02, numerical_legacy_MSE_rtol=1e-10, numerical_legacy_MSE_atol=1e-14,
        design_basis='matched_training_telemetry_and_completed_original_policy_failure',
        own_diagnostic_values_read_for_design=False, new_unit_policy_values_read_for_design=False,
        role='label_aware_exposed_SOURCE_fitting_diagnosis_only',
        current_policy_thresholds_changed=False, independent_roles_opened=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, value in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != value: raise ValueError('Registered score audit changed: '+rel)
            parent.frozen.BASE.engine.committed(ROOT/rel)
        parent.frozen.BASE.engine.committed(REG)


def summarize(rows):
    reg = json.loads(unit.REG.read_text()); expected = {(c, s, seed)
        for c, ss in reg['contexts'].items() for s in ss for seed in (17, 29, 43)}
    if len(rows) != 72 or {(r['context'], r['source'], r['seed']) for r in rows} != expected:
        raise ValueError('Complete original72-group roster required')
    names = sorted(rows[0]['scores']); count_fields = ('rows', 'queries', 'selected_rows', 'selected_unknown_rows',
        'complete_selected_queries', 'unknown_selected_queries', 'empty_selected_queries',
        'changed_rows', 'introduced_rows', 'removed_rows', 'changed_action_rows')
    moments = ('mean_score_squared_error', 'mean_signed_score_underprediction',
        'mean_harm_underprediction', 'mean_reference_bias_contribution')
    result = {}
    for name in names:
        subset = [r['scores'][name] for r in rows]; values = {}
        for field in moments:
            per_source = {}
            for r in rows:
                value = r['scores'][name][field]
                if value is not None: per_source.setdefault(r['source'], []).append(value)
            means = {s: np.mean(v, 0).tolist() for s, v in sorted(per_source.items())}
            values[field] = dict(mean=np.mean(list(means.values()), 0).tolist() if means else None,
                by_locality=means, supported_localities=len(means), required_localities=12,
                supported_views=sum(v[field] is not None for v in subset), required_views=72,
                conditional_support_only_not_full_cohort_effect=True)
        result[name] = dict(counts={k: sum(v[k] for v in subset) for k in count_fields}, moments=values,
            predicted_signed_safe_observed_positive_excess=np.sum([v['predicted_signed_safe_observed_positive_excess'] for v in subset], 0).tolist(),
            zero_true_reference_complete_queries=np.sum([v['zero_true_reference_complete_queries'] for v in subset], 0).tolist(),
            max_abs_decomposition_error=max(v['max_abs_decomposition_error'] for v in subset))
    return dict(groups=72, variants=result, all288_original_selected_MSE_controls_close=all(r['legacy_MSE_controls_close'] for r in rows),
        result_source='fresh_inference_and_label_accounting_cached_verified_choices_not_reoptimized',
        different_selected_populations_not_causal_comparison=True, no_new_model_or_policy=True,
        independent_confirmation=False, deployment_changed=False, submission_ready=False,
        metric_claim=False, seconds_claim=False, stage5c_executed=False, smc_enabled=False)


def execute(verify=False):
    registration(); cfg, values, store, banks, models, controls = unit.admission()
    complete = json.loads((unit.PUBLIC/'complete.json').read_text())
    if complete['summary_sha256'] != sha(unit.PUBLIC/'summary.json'): raise ValueError('Complete frozen unit readout required')
    once(PUBLIC/'execution_inputs.json', dict(unit_complete_sha256=sha(unit.PUBLIC/'complete.json'),
        unit_summary_sha256=sha(unit.PUBLIC/'summary.json'), original_complete_sha256=sha(parent.PUBLIC/'complete.json'),
        original_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        unit_training_freeze_sha256=sha(unit.PUBLIC/'training_freeze.json'), registration_sha256=sha(REG)))
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Completed score audit immutable')
    fitted = {}
    for ref in json.loads((unit.PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('New fit receipt changed')
        r = json.loads((ROOT/ref['path']).read_text()); cp = r['checkpoint']
        if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('New trained critic changed')
        fitted[r['context'], r['seed'], r['arm']] = cp
    if len(fitted) != 36: raise ValueError('Complete36 frozen unit critics required')
    parent.api.torch.set_num_threads(4); parent.api.torch.set_num_interop_threads(1); began = time.monotonic()
    def beat(**event):
        if time.monotonic()-began > 43200: raise TimeoutError('12h read-only audit limit')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE/'heartbeat.json', row); print(json.dumps(row), flush=True)
    beat(state='selected_score_audit_started', threads=4, workers=0)
    old, main = aggregate_groups(parent.frozen.source.parent.NAME), aggregate_groups(parent.frozen.NAME)
    readouts = dict(old=aggregate_groups(parent.NAME), new=aggregate_groups(unit.NAME))
    previous = aggregate_groups(NAME) if verify else {}; rows = []; refs = []
    prior = {Path(r['path']).name: r for r in json.loads((PUBLIC/'complete.json').read_text())['groups']} if verify else {}
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, changed = parent.frozen.BASE.prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            data, ranges, _ = parent.prepare(context, seed, packets, changed, store, models, old, main, beat)
            predictions = {}
            for mode, roster in (('old', controls), ('new', fitted)):
                for arm in cfg['arms']:
                    state = parent.api.read_checkpoint(ROOT/roster[context, seed, arm]['path'])
                    parent.api.exact(state['input_hashes'], data['input_hashes'])
                    predictions[mode, arm] = parent.api.predict(state, data['causal'])
            score_rms = np.sqrt(data['rms'][[1, 4]]**2+(.02*data['rms'][[2, 3]])**2)
            for source, packet in sorted(packets.items()):
                key = f'{context}_{source}_seed{seed}_fit.json'; sl = ranges[source]; scores = {}
                groups = parent.decoder.base.query_groups(packet['recordings'], packet['frames'], packet['ids'])
                initial = data['causal']['initial'][sl]
                for mode, source_run in (('old', parent), ('new', unit)):
                    choices = parent.api.read_checkpoint(source_run.PRIVATE/'progress'/(key+'.pt.gz'))['choices']
                    result = readouts[mode][key]
                    for prefix, arm in (('row', 'row_local'), ('set', 'selected_set')):
                        pred = predictions[mode, arm][sl]; choice = choices[prefix+'_joint']
                        if (parent.frozen.BASE.engine.fingerprint(pred) != result['prediction_sha256'][arm]
                                or parent.frozen.BASE.engine.fingerprint(choice) != result['action_sha256'][prefix+'_joint']):
                            raise ValueError('Frozen critic inference or policy identity changed')
                        for scope, c in (('original', initial), ('decoded', choice)):
                            scores[mode+'_'+prefix+'_'+scope] = audit(pred, packet['y'], packet['scales'], packet['allowed'], groups, initial, c)
                        value = scores[mode+'_'+prefix+'_original']['mean_score_squared_error']
                        actual = float(np.mean(np.asarray(value)/score_rms**2)) if value is not None else None
                        expected = result['result']['calibration'][arm]['frozen_selected_query_score_MSE']
                        if ((actual is None) != (expected is None)
                                or actual is not None and not np.isclose(actual, expected, rtol=1e-10, atol=1e-14)):
                            raise ValueError('Original selected-query MSE control differs beyond declared rounding tolerance')
                row = dict(context=context, source=source, seed=seed, scores=scores, legacy_MSE_controls_close=True,
                    ids_sha256=parent.frozen.BASE.engine.fingerprint(packet['ids']), registration_sha256=sha(REG),
                    execution_inputs_sha256=sha(PUBLIC/'execution_inputs.json'), no_new_model_or_policy=True)
                path = PUBLIC/'groups'/key
                if verify:
                    if row != previous[key]: raise ValueError('Separate complete score audit differs')
                else: once(path, row)
                rows.append(row); refs.append(prior[key] if verify else dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='selected_score_group_complete', groups=len(rows), total=72, context=context, source=source, seed=seed)
            del data, predictions
        del packets, changed
    summary = summarize(rows)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Complete score summary differs')
        once(PUBLIC/'verification.json', dict(all72groups_exact=True, summary_sha256=sha(PUBLIC/'summary.json'), complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-began,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'), execution_inputs_sha256=sha(PUBLIC/'execution_inputs.json')))
        pack(ROOT, PUBLIC)
    once(PUBLIC/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-began, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72_selected_score_groups_verified' if verify else 'all72_selected_score_groups_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'run', 'verify'))
    a = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration(register=True) if a.phase == 'register' else execute(a.phase == 'verify')
