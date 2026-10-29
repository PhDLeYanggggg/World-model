"""Audit frozen factorial receipts and report without new inference or fitting."""
import argparse
from collections import Counter
import json
import statistics

from scripts import evaluate_m3w_decoder_risk_factorial as evaluation

run, api = evaluation.run, evaluation.api
equal = evaluation.equal_number


def check_roster(reg, documents):
    expected = {(context, seed, arm) for context in reg['contexts']
                for seed in reg['seeds'] for arm in api.ARMS}
    if set(documents) != expected:
        raise ValueError('Exactly the registered context/seed/arm roster is required')
    for (context, seed, arm), doc in documents.items():
        partitions = {i['packet']['identity']['source']: i['partition'] for i in reg['contexts'][context]}
        if doc['identity'] != dict(context=context, seed=seed, partitions=partitions):
            raise ValueError('Fit receipt partition differs from registration')
        expected_count = 13644 if arm.startswith('source_') else 13149
        if (doc['parameter_count'] != expected_count or doc['primary_parameter_count'] != expected_count-792
                or doc['previous_control_exact'] != (True if arm == 'shared_risk' else None)):
            raise ValueError('Model capacity or exact-control declaration changed')


def check_view(row, partition):
    if row['scope'] not in ('fit', 'held'):
        raise ValueError('Unknown evaluation scope')
    result = row['result']
    if set(result['recordings']) != set(partition[row['scope']+'_recordings']):
        raise ValueError('Readout differs from the frozen whole-recording partition')
    if any(row[k] for k in ('outer_development_read', 'independent_roles_read', 'deployment_changed')):
        raise ValueError('Evidence scope changed')
    evaluation.check_result(result)
    # These masses are not all covered by the original readout sum check.
    for arm in evaluation.ARMS:
        for key in ('selected_known_easy_benefit_mass', 'full_known_easy_reference_mass'):
            equal(result['policies'][arm][key], sum(r[arm][key] for r in result['recordings'].values()))
        equal(result['policies'][arm]['known_rows'], result['known_rows'])
        equal(result['policies'][arm]['unknown_rows'], result['unknown_rows'])
    a = result['contrasts']['source_risk_minus_shared_risk_MSE']
    b = result['contrasts']['source_utility_minus_shared_utility_MSE']
    equal(result['contrasts']['decoder_by_risk_MSE_interaction'], a-b if a is not None and b is not None else None)


def details(rows):
    result = {}
    for scope in ('fit', 'held'):
        group = [r for r in rows if r['scope'] == scope]
        result[scope] = {}
        for arm in evaluation.ARMS:
            policies = [r['result']['policies'][arm] for r in group]
            mse = [r['result']['scores'][arm] for r in group if r['result']['scores'][arm] is not None]
            ratios = [r['result']['diagnostics'][arm]['predicted_observed_easy_harm_ratio'] for r in group]
            ratios = [v for v in ratios if v is not None]
            result[scope][arm] = dict(mean_view_signed_MSE=statistics.mean(mse) if mse else None,
                defined_MSE_views=len(mse), empty_actions=sum(p['selected_count'] == 0 for p in policies),
                unknown_selected_occurrences=sum(p['selected_unknown'] for p in policies),
                net_easy_upper_violations=sum(p['easy_degradation_upper'] is not None
                    and p['easy_degradation_upper'] > .02+1e-12 for p in policies),
                median_predicted_observed_easy_harm_ratio=statistics.median(ratios) if ratios else None,
                defined_easy_harm_ratios=len(ratios),
                reason_counts=dict(Counter(reason for p in policies for reason in p['reasons'])))
    result['sources'] = {}
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source'] == source and r['scope'] == 'held']
        result['sources'][source] = {arm: dict(
            known_easy_violations=sum(r['result']['policies'][arm]['known_easy_positive_risk'] is not None
                and r['result']['policies'][arm]['known_easy_positive_risk'] > .02+1e-12 for r in group),
            full_support=sum(r['result']['policies'][arm]['finite_completion_supported'] for r in group),
            views=len(group)) for arm in evaluation.ARMS}
    return result


def validate():
    evaluation.execute(check=True)
    reg, cfg, _ = run.admission()
    freeze, documents = evaluation.models()
    check_roster(reg, documents)
    if (freeze['optimizer_steps'] != 144000 or freeze['exact_previous_controls'] != 18
            or not freeze['matched_sampling'] or freeze['inner_held_scored']
            or any(freeze[k] for k in ('outer_development_read', 'independent_roles_read', 'deployment_changed'))):
        raise ValueError('Training freeze changed')
    controls = run.previous_eval.frozen_models()
    final_loss = {arm: [] for arm in api.ARMS}
    for context in reg['contexts']:
        for seed in reg['seeds']:
            states = [api.old.core.read_checkpoint(run.ROOT/documents[context, seed, arm]['checkpoint']['path'])
                      for arm in api.ARMS]
            api.assert_matched(states)
            old = api.old.core.read_checkpoint(run.ROOT/controls[context, seed, 'shared_source', None]['checkpoint']['path'])
            api.assert_previous_control(states[0], old)
            for state in states:
                arm = state['arm']
                doc = documents[context, seed, arm]
                if (state['experiment_sha256'] != run.base.sha(run.REG) or state['step'] != 2000
                        or state['new_forecaster'] or state['inner_held_scored']):
                    raise ValueError('Changed checkpoint evidence boundary')
                api.old.core.exact(state['identity'], doc['identity'])
                api.old.core.exact(state['trace'][-1], doc['final_trace'])
                if arm.endswith('_utility') and bool((state['dual'] != 0).any()):
                    raise ValueError('Utility arm must have zero risk multipliers')
                final_loss[arm].append(state['trace'][-1])
    complete = json.loads((evaluation.PUBLIC/'complete.json').read_text())
    rows = []
    for ref in complete['groups']:
        row = json.loads((run.ROOT/ref['path']).read_text())
        partition = next(i['partition'] for i in reg['contexts'][row['context']]
                         if i['packet']['identity']['source'] == row['source'])
        check_view(row, partition)
        rows.append(row)
    detail = details(rows)
    detail['final_sampled_training_losses'] = {
        arm: {key: statistics.mean(t[key] for t in traces)
              for key in ('loss', 'primary_loss', 'regret', 'mean_soft_action')}
        for arm, traces in final_loss.items()}
    detail.update(registered_fits=len(documents), verified_views=len(rows),
        source_localities=len(detail['sources']), fresh_training=False, fresh_inference=False,
        independent_roles_read=False, deployment_changed=False)
    return freeze, complete, json.loads((evaluation.PUBLIC/'summary.json').read_text()), detail


def interval(value):
    if value['CI95'] is None:
        return 'not estimable'
    low, high = value['CI95']
    return f"{value['mean']:+.6f} [{low:+.6f}, {high:+.6f}]"


def report(freeze, complete, summary, detail):
    lines = ['# Source Decoder by Risk Objective: Complete Fixed-Final Readout', '',
        '## Execution and Provenance', '',
        f"Fresh_run:72 cost-head fits,144,000 updates in {freeze['seconds']:.2f}s;144 fit/held readouts "
        f"in {complete['seconds']:.2f}s. All72 finals were frozen before inference.",
        'The18 shared-risk models reproduce their old controls exactly: weights, optimizer,',
        'duals, traces and samplers. Cached_verified source-local heads are unchanged.',
        'Across all four arms, source queries, preprocessing and initial predictions match.',
        'Four source decoders add495 parameters; this is not a parameter-matched test.',
        'No new trajectory producer, JEPA representation, residual or latent rollout ran.', '',
        'The head predicts benefit B, positive harm H, floor-reference R, easy',
        'reference R_easy and easy harm H_easy. Signed scores are B-H, H-0.02R',
        'and H_easy-0.02R_easy. Reported score MSE is not trajectory FDE/ADE.', '',
        'Original source TRAIN is partitioned by whole recording. Both fit and held',
        'recordings remain TRAIN-internal; held data have prior research exposure.',
        'No outer development or independent selection/calibration/confirmation was read.',
        'Preprocessing uses inner-fit only. Unknown outcomes remain in inference and',
        'completion bounds; future supervision is not an inference feature.', '']
    for scope in ('fit', 'held'):
        s = summary['scopes'][scope]
        lines += [f'## {scope.title()} Recording Results', '',
            f"{s['rows_per_arm']:,} repeated row occurrences/arm, including {s['unknown_rows_per_arm']:,} "
            'unknown outcomes. Occurrences overlap across context/head-seed views.', '',
            '| Contrast | Full lower utility, pp [95% CI] | Same-count lower utility, pp [CI] | Signed-score MSE difference [CI] |',
            '|---|---:|---:|---:|']
        for left, right in evaluation.PAIRS:
            key = right+'_minus_'+left
            lines.append('| '+right+' minus '+left+' | '+' | '.join(
                interval(s['contrasts'][key+'_'+kind]) for kind in ('full', 'matched', 'MSE'))+' |')
        lines += ['', 'Decoder-by-risk MSE interaction: '+interval(s['contrasts']['decoder_by_risk_MSE_interaction'])+'.', '',
            '| Arm | Selected | Known easy violations /72 | Upper easy violations /72 | Undefined easy /72 | Full support /72 | Mean view MSE |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for arm in evaluation.ARMS:
            p = s['safety'][arm]
            vals = [str(p[k]) for k in ('selected_occurrences', 'known_easy_violations',
                'easy_upper_violations', 'undefined_easy', 'full_completion_support')]
            vals.append(f"{detail[scope][arm]['mean_view_signed_MSE']:.6f}")
            lines.append('| '+arm+' | '+' | '.join(vals)+' |')
    lines += ['', '## Interpretation Boundary', '',
        'Utility is a pair-specific completion-lower difference divided by full known',
        'floor-reference error mass, not ADE/FDE improvement. Shared unknown actions',
        'cancel in a comparison; unmatched unknown actions retain worst-case bounds.',
        'Pair-specific lower utilities are not transitive and must not be subtracted',
        'to create an interaction. Only the MSE difference-in-differences is reported.',
        'The nominal3,000-draw bootstrap averages repeated views within12 localities.',
        'Shared training, reused held recordings and multiple contrasts limit inference:',
        'these are exposed TRAIN-internal intervals, not independent significance.',
        'Three cost-head seeds are not three independent forecasting-model seeds.', '',
        'The2% positive-harm/reference evaluation budget is identical in every arm.',
        'Removing a training penalty does not relax safety. Undefined/empty risk is',
        'not a pass. Full support additionally requires easy preservation and positive',
        'completion utility; same-count risk is diagnostic, not a second deployment gate.', '',
        'Final sampled training losses are retained in verified_details.json.',
        'The risk objective includes a signed Lagrangian penalty and can be negative;',
        'its total loss is not directly comparable with the utility-only total.',
        'Last-batch loss is not a full-training-census error. Easy-harm calibration',
        'ratios also use different selected cohorts in different arms, not a controlled',
        'fixed-cohort calibration comparison.', '',
        '## Registered Routing Decision', '']
    for arm, outcome in summary['routing'].items():
        lines += [f"- {arm}: eligible for later development design = {outcome['eligible_for_later_development_design']}.",
                  '  Reasons: '+(', '.join(outcome['failures']) or 'all prespecified TRAIN-internal conditions met')+'.']
    lines += ['', 'No deployment changed. A routing pass would still not authorize an',
        'independent-test claim or establish M3W/CVPR success.', '',
        '## Held Source Slices', '',
        '| Locality | Local | Shared risk | Shared utility | Source risk | Source utility |',
        '|---|---:|---:|---:|---:|---:|']
    for source, arms in detail['sources'].items():
        lines.append('| '+source+' | '+' | '.join(
            f"{arms[a]['known_easy_violations']}/{arms[a]['views']}" for a in evaluation.ARMS)+' |')
    lines += ['', 'Cells show known easy-risk violations. The JSON retains undefined risk,',
        'completion support, unknown selections and calibration diagnostics separately.', '',
        '## Reproduce and Inspect', '', '```bash',
        '.venv-pytorch/bin/python -m scripts.evaluate_m3w_decoder_risk_factorial check',
        '.venv-pytorch/bin/python -m scripts.report_m3w_decoder_risk_factorial --check', '```', '',
        'Completed artifacts are immutable; an interrupted unfinished phase uses its',
        'explicit --resume option. Checkpoints remain local and excluded from Git.',
        'This verifier runs no optimizer and opens no new data role.', '',
        'Observation8/prediction12, raw-frame stride12, dataset-local coordinates.',
        'Historical t+50 remains separate. No metric/seconds, human-gold, true3D,',
        'foundation or physical-safety claim. Stage5C and SMC remain disabled.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    freeze, complete, summary, detail = validate()
    rendered = report(freeze, complete, summary, detail)
    path = evaluation.PUBLIC/'conclusions.md'
    output = evaluation.PUBLIC/'verified_details.json'
    if args.check:
        if path.read_text() != rendered or json.loads(output.read_text()) != detail:
            raise ValueError('Report differs from verified evidence')
        print('Verified full roster, source partitions,72 checkpoints,18 exact controls and144 readouts')
    else:
        run.base.once(output, detail)
        if path.exists() and path.read_text() != rendered:
            raise ValueError('Do not overwrite a completed report')
        path.write_text(rendered)
        print(path)


if __name__ == '__main__':
    main()
