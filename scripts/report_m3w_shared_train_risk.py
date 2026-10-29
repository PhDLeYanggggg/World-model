"""Verify frozen shared-risk evidence without opening any new data role."""
import argparse
from collections import Counter
import json
import math
import statistics

from scripts import evaluate_m3w_shared_train_risk as evaluation

run = evaluation.run
api = evaluation.api


def equal_number(actual, expected):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError('Missing denominator must remain undefined')
    elif not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError('Aggregate arithmetic does not replay')


def check_policy(p):
    benefit = p['selected_known_benefit_mass']
    harm = p['selected_known_harm_mass']
    unknown = p['selected_unknown_envelope_mass']
    reference = p['selected_known_reference_mass']
    easy_reference = p['selected_known_easy_reference_mass']
    easy_harm = p['selected_known_easy_harm_mass']
    easy_benefit = p['selected_known_easy_benefit_mass']
    full_easy = p['full_known_easy_reference_mass']
    expected = {
        'selected_net_gain_lower_mass': benefit-harm-unknown,
        'all_budget_slack_mass': .02*reference-harm,
        'easy_budget_slack_mass': .02*easy_reference-easy_harm,
        'all_selected_risk_upper': (harm+unknown)/reference if reference > 0 else None,
        'easy_selected_risk_upper': (easy_harm+unknown)/easy_reference if easy_reference > 0 else None,
        'known_positive_risk': harm/reference if reference > 0 else None,
        'known_easy_positive_risk': easy_harm/easy_reference if easy_reference > 0 else None,
        'switch_rate': p['selected_count']/(p['known_rows']+p['unknown_rows'])
            if p['known_rows']+p['unknown_rows'] else None,
    }
    numerator = easy_harm-easy_benefit+unknown
    if p['unknown_rows']:
        numerator = max(numerator, 0.)
    expected['easy_degradation_upper'] = numerator/full_easy if full_easy > 0 else None
    for key, value in expected.items():
        equal_number(p[key], value)
    reasons = []
    if expected['all_selected_risk_upper'] is None or expected['easy_selected_risk_upper'] is None:
        reasons.append('selected_reference_undefined')
    for key, reason in (
            ('all_selected_risk_upper', 'all_positive_harm_upper_exceeds_budget'),
            ('easy_selected_risk_upper', 'easy_positive_harm_upper_exceeds_budget')):
        if expected[key] is not None and expected[key] > .02+1e-12:
            reasons.append(reason)
    if expected['selected_net_gain_lower_mass'] <= 0:
        reasons.append('positive_completion_utility_not_supported')
    if expected['easy_degradation_upper'] is None or expected['easy_degradation_upper'] > .02+1e-12:
        reasons.append('easy_preservation_not_supported')
    if (p['reasons'] != reasons or p['finite_completion_supported'] != (not reasons)
            or p['per_row_label_inference_gate'] or p['population_safety_guarantee']):
        raise ValueError('Changed completion or safety contract')


def check_result(result):
    if result['rows'] != result['known_rows']+result['unknown_rows'] or result['inference_label_filter']:
        raise ValueError('All inference rows must remain present')
    masses = ('selected_unknown_envelope_mass', 'selected_known_benefit_mass',
              'selected_known_harm_mass', 'selected_known_reference_mass',
              'selected_known_easy_reference_mass', 'selected_known_easy_harm_mass',
              'selected_known_easy_benefit_mass', 'full_known_easy_reference_mass')
    for arm in evaluation.ARMS:
        policy = result['policies'][arm]
        check_policy(policy)
        for p in result['recordings'].values():
            check_policy(p[arm])
        for key in masses+('known_rows', 'unknown_rows', 'selected_count', 'selected_unknown'):
            equal_number(policy[key], sum(p[arm][key] for p in result['recordings'].values()))
    for left, right in evaluation.PAIRS:
        contrast = right+'_minus_'+left
        a, b = result['scores'][left], result['scores'][right]
        equal_number(result['contrasts'][contrast+'_MSE'], b-a if a is not None and b is not None else None)
        matched = result['matched_policies'][contrast]
        for p in matched.values():
            check_policy(p)
        if matched[left]['selected_count'] != matched[right]['selected_count']:
            raise ValueError('Same-count comparison changed intervention count')
        for mode in ('full', 'matched'):
            pair = result['pairs'][contrast+'_'+mode]
            equal_number(pair['lower_mass'], pair['known_difference_mass']-pair['unknown_exchanged_envelope_mass'])
            equal_number(pair['upper_mass'], pair['known_difference_mass']+pair['unknown_exchanged_envelope_mass'])
            den = result['full_known_reference_mass']
            equal_number(result['contrasts'][contrast+'_'+mode], 100*pair['lower_mass']/den if den > 0 else None)


def validate():
    reg, cfg, _ = run.admission()
    public = evaluation.PUBLIC
    read_reg = json.loads(evaluation.REG.read_text())
    for rel, digest in read_reg['bindings'].items():
        if run.base.sha(run.ROOT/rel) != digest:
            raise ValueError('Readout implementation differs from registration')
    if read_reg['training_registration_sha256'] != run.base.sha(run.REG):
        raise ValueError('Readout no longer refers to registered training')
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    complete = json.loads((public/'complete.json').read_text())
    summary = json.loads((public/'summary.json').read_text())
    if (freeze['optimizer_steps'] != 216000 or freeze['inner_held_scored']
            or not freeze['matched_per_source_draws']
            or complete['summary_sha256'] != run.base.sha(public/'summary.json')
            or complete['registration_sha256'] != run.base.sha(evaluation.REG)
            or complete['training_freeze_sha256'] != run.base.sha(run.PUBLIC/'training_freeze.json')
            or any(d[k] for d in (freeze, complete)
                   for k in ('outer_development_read', 'independent_roles_read', 'deployment_changed'))):
        raise ValueError('Freeze or data-role boundary changed')
    models = evaluation.frozen_models()
    expected = set()
    for context, items in reg['contexts'].items():
        sources = sorted(item['packet']['identity']['source'] for item in items)
        for seed in reg['seeds']:
            expected.update((context, seed, 'local', source) for source in sources)
            expected.update((context, seed, arm, None) for arm in evaluation.ARMS[1:])
            states = {arm: api.old.core.read_checkpoint(run.ROOT/models[context, seed, arm, None]['checkpoint']['path'])
                      for arm in evaluation.ARMS[1:]}
            for state in states.values():
                if (state['step'] != 2000 or state['experiment_sha256'] != run.base.sha(run.REG)
                        or state['inner_held_scored'] or state['new_forecaster']):
                    raise ValueError('Shared checkpoint provenance changed')
            for key in ('preprocess', 'source_preprocess', 'initial_model', 'input_hashes', 'samplers'):
                api.old.core.exact(states['shared_global'][key], states['shared_source'][key])
            for source in sources:
                local = api.old.core.read_checkpoint(run.ROOT/models[context, seed, 'local', source]['checkpoint']['path'])
                sampler = states['shared_source']['samplers'][source]
                api.old.core.exact(sampler['rng'], local['sampler_rng'])
                api.old.core.exact(states['shared_source']['source_preprocess'][source], local['preprocess'])
                api.old.core.exact(states['shared_source']['input_hashes'][source], local['input_hashes'])
                if local['step'] != 2000 or local['experiment_sha256'] != run.base.sha(run.REG):
                    raise ValueError('Local checkpoint provenance changed')
                if sampler['draw_hash'] != local['draw_hash'] or sampler['row_draws'] != local['row_draws']:
                    raise ValueError('Source exposure differs between local and pooled heads')
    if set(models) != expected:
        raise ValueError('Complete registered 108-fit roster required')
    rows = []
    for ref in complete['groups']:
        path = run.ROOT/ref['path']
        if (path.is_symlink() or not path.resolve().is_relative_to((public/'groups').resolve())
                or run.base.sha(path) != ref['sha256']):
            raise ValueError('Readout group changed')
        row = json.loads(path.read_text())
        if (row['readout_registration_sha256'] != run.base.sha(evaluation.REG)
                or row['training_freeze_sha256'] != run.base.sha(run.PUBLIC/'training_freeze.json')
                or not row['exact_inference_replay'] or row['outer_development_read']
                or row['independent_roles_read'] or row['deployment_changed']):
            raise ValueError('Readout provenance changed')
        item = next(item for item in reg['contexts'][row['context']]
                    if item['packet']['identity']['source'] == row['source'])
        if set(row['result']['recordings']) != set(item['partition']['held_recordings']):
            raise ValueError('Readout recordings differ from frozen inner holdout')
        check_result(row['result'])
        rows.append(row)
    if evaluation.summarize(rows, reg, cfg) != summary:
        raise ValueError('Complete readout summary does not replay')
    detail = dict(views=len(rows), fits=len(models), optimizer_steps=216000,
        rows_per_arm=sum(r['result']['rows'] for r in rows),
        unknown_rows_per_arm=sum(r['result']['unknown_rows'] for r in rows),
        source_localities=len({r['source'] for r in rows}),
        partitions_with_one_fit_recording=sum(len(item['partition']['fit_recordings']) == 1
            for items in reg['contexts'].values() for item in items),
        policy_details={}, new_training=False, new_inference=False,
        outer_development_read=False, independent_roles_read=False, deployment_changed=False)
    for arm in evaluation.ARMS:
        policies = [r['result']['policies'][arm] for r in rows]
        ratio = [r['result']['diagnostics'][arm]['predicted_observed_easy_harm_ratio'] for r in rows]
        defined = [v for v in ratio if v is not None]
        detail['policy_details'][arm] = dict(
            failure_reason_counts=dict(Counter(reason for p in policies for reason in p['reasons'])),
            unknown_selected_occurrences=sum(p['selected_unknown'] for p in policies),
            median_predicted_observed_easy_harm_ratio=statistics.median(defined) if defined else None,
            defined_easy_harm_ratios=len(defined),
            known_easy_pass_but_completion_easy_fails=sum(
                p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] <= .02+1e-12
                and p['easy_selected_risk_upper'] is not None and p['easy_selected_risk_upper'] > .02+1e-12
                for p in policies))
    detail['source_breakdown'] = {}
    for source in sorted({r['source'] for r in rows}):
        local_rows = [r for r in rows if r['source'] == source]
        detail['source_breakdown'][source] = dict(
            repeated_views=len(local_rows),
            completion_support={arm: sum(r['result']['policies'][arm]['finite_completion_supported']
                                        for r in local_rows) for arm in evaluation.ARMS},
            known_easy_violations={arm: sum(
                r['result']['policies'][arm]['known_easy_positive_risk'] is not None
                and r['result']['policies'][arm]['known_easy_positive_risk'] > .02+1e-12
                for r in local_rows) for arm in evaluation.ARMS})
    return freeze, complete, summary, detail


def report(freeze, complete, summary, detail):
    lines = ['# Shared TRAIN Risk: Fixed Recording-Transfer Evidence', '',
        '## What Ran', '',
        f"Fresh_run: {freeze['fresh_fits']} cost heads, {freeze['optimizer_steps']:,} optimizer updates, "
        f"{freeze['seconds']:.2f}s fitting and {complete['seconds']:.2f}s fixed-final inference.",
        'The 72 local heads and 18 heads in each shared arm have identical per-source',
        'query draws. Shared arms have identical initialization. Checkpoints and',
        'all final receipts are hash-verified; held predictions replay exactly.',
        'Source TRAIN packets and frozen forecasting models are cached_verified.',
        'There is no newly trained trajectory model, JEPA representation or residual.', '',
        f"Each arm has {detail['rows_per_arm']:,} held-recording row occurrences, including "
        f"{detail['unknown_rows_per_arm']:,} wholly unknown outcomes. These repeat across",
        'contexts/seeds and are not independent samples. Preprocessing and fitting',
        'use inner-fit recordings only; original outer development and independent',
        'selection/calibration/confirmation roles were not opened by this experiment.', '',
        '## Paired Comparisons', '',
        '| Shared repair minus control | Full lower utility, pp (95% CI) | Same-count lower utility, pp (95% CI) | Signed-score MSE difference (95% CI) |',
        '|---|---:|---:|---:|']
    def number(value):
        if value['CI95'] is None:
            return 'not estimable'
        low, high = value['CI95']
        return f"{value['mean']:+.6f} [{low:+.6f}, {high:+.6f}]"
    for left, right in evaluation.PAIRS:
        key = right+'_minus_'+left
        lines.append('| '+right+' minus '+left+' | '+' | '.join(
            number(summary['contrasts'][key+'_'+suffix]) for suffix in ('full', 'matched', 'MSE'))+' |')
    lines += ['', 'Utility is the pair-specific completion lower bound, divided by full',
        'known floor-reference error mass. It is not FDE improvement. Shared unknown',
        'actions cancel; unmatched unknown actions retain worst-case disagreement.',
        'The 3,000-draw nominal bootstrap averages repeated views inside each of 12',
        'source localities. Shared training and prior source exposure prevent an',
        'independent significance or population safety claim. Head seeds are not',
        'independent forecasting seeds.', '', '## Safety and Support', '',
        '| Arm | Selected occurrences | Known easy violations /72 | Completion-upper easy violations /72 | Undefined easy risk /72 | Full completion support /72 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in evaluation.ARMS:
        s = summary['safety'][arm]
        lines.append('| '+arm+' | '+' | '.join(str(s[k]) for k in (
            'selected_occurrences', 'known_easy_violations', 'easy_upper_violations',
            'undefined_easy_risk', 'completion_supported'))+' |')
    lines += ['', 'The unchanged 2% budget concerns positive selected harm/reference, not',
        'net easy ADE. No missing label is imputed safe. Empty same-count risk remains',
        'undefined and is reported separately from full-policy safety.', '',
        '## Source Breakdown', '',
        '| Source | Repeated views | Known easy violations: local / shared-global / shared-source | Completion support: local / shared-global / shared-source |',
        '|---|---:|---:|---:|']
    for source, row in detail['source_breakdown'].items():
        lines.append('| '+source+' | '+str(row['repeated_views'])+' | '+' | '.join(
            ' / '.join(str(row[key][arm]) for arm in evaluation.ARMS)
            for key in ('known_easy_violations', 'completion_support'))+' |')
    lines += ['', 'These source-level slices include every registered view. They are diagnosis,',
        'not permission to drop a failing source or choose a deployment threshold.', '', '## Decision', '']
    for arm, value in summary['routing'].items():
        lines.append(f"- {arm}: eligible for later development design = "
                     f"{value['eligible_for_later_development_design']}; failures: "
                     +(', '.join(value['failure_reasons']) or 'none')+'.')
    lines += ['', 'Neither this inner-TRAIN comparison nor routing eligibility authorizes',
        'deployment or independent-test access. Negative controls remain visible.',
        'Pooling changes the learned representation and prior as well as parameter',
        'sharing; only shared-source versus shared-global isolates dual grouping.',
        'Optimization loss contains a Lagrangian and can be negative; it is not a',
        'negative ADE and is not used to declare success.', '', '## Reproduction', '',
        '```bash', '.venv-pytorch/bin/python -m scripts.report_m3w_shared_train_risk --check',
        '```', '', 'This rechecks all 108 weights, exact matched sampling, 72 readout receipts,',
        'recording partitions, risk ratios, aggregate sums and bootstrap summary.',
        'No new training or role access is required. Observation8/prediction12,',
        'dataset-local coordinates and raw-frame stride12 remain unchanged. Not',
        'true3D, not foundation, no metric/seconds claim; Stage5C and SMC disabled.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    freeze, complete, summary, detail = validate()
    artifacts = {'conclusions.md': report(freeze, complete, summary, detail),
                 'verified_details.json': json.dumps(detail, indent=2, allow_nan=False)+'\n'}
    for name, text in artifacts.items():
        path = evaluation.PUBLIC/name
        if args.check or path.exists():
            if path.read_text() != text:
                raise ValueError('Preserve conflicting report')
        else:
            with path.open('x') as stream:
                stream.write(text)
    print(json.dumps(dict(status='checked' if args.check else 'reported', fits=108,
                          views=72, new_training=False, new_inference=False)))


if __name__ == '__main__':
    main()
