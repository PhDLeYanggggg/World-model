"""Verify and render the immutable protected-harm experiment."""
import argparse
from collections import Counter
import json
import statistics

from scripts import evaluate_m3w_isolated_harm as evaluation

run, api = evaluation.run, evaluation.api


def details(rows):
    result = {}
    for scope in ('fit', 'held'):
        group = [r for r in rows if r['scope'] == scope]
        result[scope] = {}
        for arm in evaluation.ARMS:
            policies = [r['result']['policies'][arm] for r in group]
            scores = [r['result']['scores'][arm] for r in group if r['result']['scores'][arm] is not None]
            d = dict(mean_view_signed_MSE=statistics.mean(scores) if scores else None,
                defined_MSE_views=len(scores), unknown_selected_occurrences=sum(p['selected_unknown'] for p in policies),
                reason_counts=dict(Counter(reason for p in policies for reason in p['reasons'])))
            for cohort in ('diagnostics', 'fixed_anchor_cohort'):
                ratios = [r['result'][cohort][arm]['predicted_observed_easy_harm_ratio'] for r in group]
                ratios = [v for v in ratios if v is not None]
                d[cohort] = dict(median_predicted_observed_easy_harm_ratio=statistics.median(ratios) if ratios else None,
                    defined_ratios=len(ratios), known_selected_occurrences=sum(r['result'][cohort][arm]['known_selected'] for r in group))
            result[scope][arm] = d
    result['sources'] = {}
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['scope'] == 'held' and r['source'] == source]
        result['sources'][source] = {arm: dict(views=len(group),
            selected=sum(r['result']['policies'][arm]['selected_count'] for r in group),
            known_easy_violations=sum(r['result']['policies'][arm]['known_easy_positive_risk'] is not None
                and r['result']['policies'][arm]['known_easy_positive_risk'] > .02+1e-12 for r in group),
            full_support=sum(r['result']['policies'][arm]['finite_completion_supported'] for r in group))
            for arm in evaluation.ARMS}
    return result


def validate():
    evaluation.execute(check=True)
    reg, _, _ = run.admission()
    freeze, docs = evaluation.models(reg)
    traces = {a: [] for a in api.ARMS}
    for (_, _, arm), doc in docs.items():
        expected_active = 12258 if arm == 'isolated_risk' else 24615
        if doc['total_parameters'] != 25407 or doc['trainable_parameters'] != expected_active:
            raise ValueError('Unexpected continuation capacity')
        state = api.core.read_checkpoint(run.ROOT/doc['checkpoint']['path'])
        api.core.exact(doc['final_trace'], state['trace'][-1])
        api.core.exact(doc['anchor'], dict(path=doc['anchor']['path'], sha256=state['anchor_sha256'], bytes=doc['anchor']['bytes']))
        if state['new_forecaster'] or state['inner_held_scored']:
            raise ValueError('Checkpoint changed role')
        traces[arm].append(state['trace'][-1])
    complete = json.loads((evaluation.PUBLIC/'complete.json').read_text())
    rows = [json.loads((run.ROOT/ref['path']).read_text()) for ref in complete['groups']]
    detail = details(rows)
    recording_sets = {scope: {(i['packet']['identity']['source'], recording)
        for items in reg['contexts'].values() for i in items for recording in i['partition'][scope+'_recordings']}
        for scope in ('fit', 'held')}
    if recording_sets['fit'] & recording_sets['held']:
        raise ValueError('Recording crosses inner-fit and held roles')
    detail['distinct_source_recordings'] = {k: len(v) for k, v in recording_sets.items()}
    detail['final_sampled_training_losses'] = {arm: {key: statistics.mean(t[key] for t in values)
        for key in ('loss', 'primary_loss', 'regret', 'mean_soft_action')} for arm, values in traces.items()}
    detail.update(verified_fits=54, verified_views=144, source_localities=len(detail['sources']),
        fresh_training=False, fresh_inference=False, independent_roles_read=False, deployment_changed=False)
    return freeze, complete, json.loads((evaluation.PUBLIC/'summary.json').read_text()), detail


def interval(v):
    if v['CI95'] is None:
        return 'not estimable'
    return f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"


def number(v):
    return 'undefined' if v is None else f'{v:.6f}'


def report(freeze, complete, summary, detail):
    lines = ['# Protected Harm Continuation: Complete Readout', '',
        '## Execution and Evidence', '',
        f"Fresh_run:54 continuations/108,000 new updates in{freeze['seconds']:.2f}s;144 "
        f"fit/held views in{complete['seconds']:.2f}s after all finals were frozen.",
        'All three arms start from the same shared-utility anchors and continue identical',
        'source/query draws with fresh optimizers. Each continued head has4,000 effective',
        'updates including its2,000-update anchor. Local, old-risk and anchor controls',
        'remain cached_verified2,000-update controls and reproduce their old inference.',
        'Utility continuation and joint risk are the equal-additional-budget controls.',
        'Total parameters25,407 in all arms; isolated harm trains12,258 versus24,615',
        'in the joint/utility arms. This is not an equal-trainable-capacity comparison.', '',
        'Benefit B, positive harm H, floor error R, easy floor R_easy and easy harm',
        'H_easy are cost moments. Signed scores are B-H, H-0.02R and H_easy-0.02R_easy.',
        'Isolated B/R/R_easy remain bit-exact to the anchor on every evaluated row.',
        'H changes, so net gain is not frozen. No trajectory predictor was trained.', '',
        'Original TRAIN is split by whole recording; held recordings were previously',
        'exposed. This is exploratory mechanism evidence, not independent confirmation.',
        f"Distinct source/recording pairs: {detail['distinct_source_recordings']['fit']} fit and "
        f"{detail['distinct_source_recordings']['held']} held, across12 source localities.",
        'No outer development, independent selection/calibration/test or deployment',
        'changed. Unknown labels remain eligible at inference and receive bounds.', '']
    for scope in ('fit', 'held'):
        s = summary['scopes'][scope]
        lines += [f'## {scope.title()} Recording Results', '',
            f"{s['rows_per_arm']:,} repeated row occurrences/arm;{s['unknown_rows_per_arm']:,} unknown.", '',
            '| Arm | Mean signed MSE | Selected | Known easy violations /72 | Upper easy violations /72 | Undefined easy /72 | Full support /72 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for arm in evaluation.ARMS:
            p = s['safety'][arm]
            values = [number(detail[scope][arm]['mean_view_signed_MSE'])]+[str(p[k]) for k in
                ('selected_occurrences', 'known_easy_violations', 'easy_upper_violations', 'undefined_easy', 'full_completion_support')]
            lines.append('| '+arm+' | '+' | '.join(values)+' |')
        lines += ['', '| Contrast | Full lower utility pp [95% CI] | Same-count lower utility pp [CI] | Signed MSE difference [CI] |',
            '|---|---:|---:|---:|']
        for left, right in evaluation.PAIRS:
            key = right+'_minus_'+left
            lines.append('| '+right+' minus '+left+' | '+' | '.join(interval(s['contrasts'][key+'_'+k])
                for k in ('full', 'matched', 'MSE'))+' |')
        lines += ['', '### Same Anchor-Selected Cohort: Harm Diagnosis', '',
            '| Arm | Median predicted / observed easy harm | Defined views /72 | Known selected occurrences |',
            '|---|---:|---:|---:|']
        for arm in evaluation.ARMS:
            d = detail[scope][arm]['fixed_anchor_cohort']
            lines.append(f"| {arm} | {number(d['median_predicted_observed_easy_harm_ratio'])} | {d['defined_ratios']} | {d['known_selected_occurrences']} |")
        lines += ['', 'These ratios use exactly the same anchor-selected cohort within each view.',
            'They are aggregate harm diagnostics, not ECE or a population calibration guarantee.',
            'Own-selected ratios are retained separately in verified_details.json.', '']
    lines += ['## Registered Decision', '',
        f"Eligible for later development design: **{summary['eligible_for_later_development_design']}**.",
        'Reasons: '+(', '.join(summary['failures']) or 'All prespecified TRAIN-internal criteria met')+'.', '',
        'Every full/same-count comparison against all five controls must improve, with',
        'full-policy support in all72 held views. MSE improvements alone cannot pass.',
        'The2% positive-harm/reference budget is unchanged; empty or undefined risk is',
        'not safety. No source, seed, zero-action or missing-label case was discarded.', '',
        'Utility is the pair-specific completion-lower difference divided by full-known',
        'floor reference mass, NOT ADE/FDE improvement. Pair-specific lower bounds are',
        'not transitive. Bootstrap3,000 averages repeated views within12 localities.',
        'Intervals are nominal and exposed TRAIN-internal, not simultaneous/independent.',
        'Three cost-head seeds do not imply three newly trained forecasting producers.', '',
        'Final sampled losses are in verified_details.json. Signed Lagrangian totals',
        'may be negative and are not comparable objective scales across risk/utility',
        'arms; last-batch loss is not full-census prediction error.', '',
        '## Held Locality Easy-Risk Failures', '',
        '| Locality | Local | Anchor | Old risk | Utility continued | Joint risk | Isolated risk |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for source, arms in detail['sources'].items():
        lines.append('| '+source+' | '+' | '.join(f"{arms[a]['known_easy_violations']}/{arms[a]['views']}" for a in evaluation.ARMS)+' |')
    lines += ['', '## Reproduction', '', '```bash',
        '.venv-pytorch/bin/python -m scripts.evaluate_m3w_isolated_harm check',
        '.venv-pytorch/bin/python -m scripts.report_m3w_isolated_harm --check', '```', '',
        'Completed artifacts are immutable; interrupted incomplete runs use --resume.',
        'Verification loads frozen checkpoints but performs no fit or fresh inference.',
        'Native arm64 CPU4/interop1/workers0; checkpoints excluded from Git.', '',
        'Observation8/prediction12, raw-frame stride12, dataset-local coordinates.',
        'No metric/seconds, physical-safety, human-gold, true3D or foundation claim.',
        'Stage5C and SMC disabled. Scientific/project success remains unproven.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    freeze, complete, summary, detail = validate()
    text = report(freeze, complete, summary, detail)
    path = evaluation.PUBLIC/'conclusions.md'
    if args.check:
        if path.read_text() != text or json.loads((evaluation.PUBLIC/'verified_details.json').read_text()) != detail:
            raise ValueError('Report differs from frozen evidence')
        print('Verified54 checkpoints,144 views, immutable benefit/reference and fixed-cohort reports')
    else:
        run.base.once(evaluation.PUBLIC/'verified_details.json', detail)
        if path.exists() and path.read_text() != text:
            raise ValueError('Completed report immutable')
        path.write_text(text)
        print(path)


if __name__ == '__main__':
    main()
