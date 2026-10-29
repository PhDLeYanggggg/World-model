"""Export completed, exactly verified cost/decision evidence without re-selection."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
STUDIES = (
    'european_multiaction_cost_v1',
    'european_multiaction_record_policy_v1',
    'european_regression_deferral_v1',
    'european_multiaction_support_scale_v1',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_rows(study, summary):
    rows = []
    for scope in ('fit', 'inner_held'):
        data = summary['scopes'][scope]
        for policy, safety in data['safety'].items():
            if policy == 'floor' or policy.endswith('_moment_parent'):
                continue
            accuracy = data['accuracy'][policy]
            rows.append(dict(study=study, scope=scope, policy=policy,
                known_ADE_gain_floor_percent=accuracy['point'],
                conditional_CI95_percent=accuracy['CI95'],
                supported_record_head_views=safety['budget_record_views'],
                required_record_head_views=safety['required_record_views'],
                undefined_easy_record_head_views=safety['undefined_easy_record_views'],
                selected=safety['selected'], selected_unknown=safety['selected_unknown'],
                full_record_roster_supported=(
                    safety['budget_record_views'] == safety['required_record_views']
                    and safety['budget_views'] == safety['required_views']),
                unique_source_recordings=data['unique_source_recordings']))
    return rows


def build():
    sources, rows, fits = {}, [], {}
    for study in STUDIES:
        folder = PUBLIC/study
        complete = json.loads((folder/'complete.json').read_text())
        verification = json.loads((folder/'verification.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        if (not verification['exact_replay'] or verification['groups'] != 144
                or verification['summary_sha256'] != sha(folder/'summary.json')
                or complete['summary_sha256'] != verification['summary_sha256']
                or summary['independent_roles_opened'] or summary['deployment_changed']):
            raise ValueError('Require complete, verified development evidence; no role promotion')
        bindings = {name: sha(folder/name) for name in
                    ('summary.json', 'complete.json', 'verification.json', 'training_freeze.json')}
        if len(complete['groups']) != 144:
            raise ValueError('Complete144-view readout required')
        if (folder/'metric_bundle.json').exists():
            from src.evaluation.m3w_metric_bundle import verify
            bundle = verify(ROOT, folder)
            bindings['metric_bundle.json'] = sha(folder/'metric_bundle.json')
            if bundle['groups'] != 144:
                raise ValueError('Complete aggregate bundle required')
        else:
            for ref in complete['groups']:
                if sha(ROOT/ref['path']) != ref['sha256']:
                    raise ValueError('Scientific aggregate group changed')
        freeze = json.loads((folder/'training_freeze.json').read_text())
        if freeze['count'] != 54 or freeze['inner_held_scored']:
            raise ValueError('All54 endpoints frozen before held readout required')
        fit_rows = []
        for ref in freeze['fits']:
            path = ROOT/ref['path']
            if sha(path) != ref['sha256']:
                raise ValueError('Fit receipt changed')
            fit_rows.append(json.loads(path.read_text()))
        by_arm = {}
        for arm in ('physical_only', 'physical_neural', 'neural_only'):
            subset = [r for r in fit_rows if r['arm'] == arm]
            if len(subset) != 18:
                raise ValueError('All six contexts and three head seeds required')
            last = [r['final_trace'] for r in subset]
            by_arm[arm] = dict(fits=18, parameters=sorted({r['parameters'] for r in subset}),
                final_trace_zero_intervention_fits=sum(r.get('selected_rows') == 0 for r in last)
                    if all('selected_rows' in r for r in last) else None,
                median_final_gradient_norm=statistics.median(r['gradient_norm'] for r in last))
        fits[study] = dict(updates=freeze['optimizer_updates'], fits=54,
            seconds=freeze['seconds'], peak_RSS_bytes=freeze['peak_RSS_bytes'], arms=by_arm)
        sources[study] = bindings
        rows.extend(policy_rows(study, summary))
    return dict(result_source='cached_verified_aggregate_export_not_new_training_or_bootstrap',
        sources=sources, training=fits, policies=rows,
        primary_scope='inner_held', required_views_repeat_context_and_head_seeds=True,
        known_label_utility_not_unknown_completion_bound=True,
        conditional_intervals_not_independent_confirmation=True,
        all_required_recordings_risk_budget=.02, count_matching_utility_only=True,
        empty_risk='not_estimable_not_safe_zero', whole_pipeline_OOF=False,
        new_neural_trajectory_producer_trained=False, scene_joint_inference=False,
        deployment_changed=False, independent_roles_opened=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def render(value):
    text = ['# Multi-Action Cost and Decision Evidence', '',
        'Derived from four completed and exactly replayed SOURCE-development studies.',
        'This exporter verifies frozen aggregates; it does not train, infer trajectories,',
        're-bootstrap, select a model, or open independent evaluation roles.', '',
        '## Fixed Inner-Held Readout', '',
        'Gain is equal-locality known-label ADE improvement over the frozen floor.',
        'Intervals are conditional3,000-locality bootstrap intervals, not independent',
        'confirmation. Repeated record/head views are not independent recordings.', '',
        '| Study | Policy | Gain % [95% CI] | Supported record/head views | Undefined easy views | Interventions (unknown) |',
        '|---|---|---:|---:|---:|---:|']
    for row in value['policies']:
        if row['scope'] != 'inner_held' or row['policy'] == 'frozen_origin_binary':
            continue
        ci = row['conditional_CI95_percent']
        gain = row['known_ADE_gain_floor_percent']
        point = 'not_estimable' if gain is None else f'{gain:.6f}'
        interval = '' if ci is None else f' [{ci[0]:.6f}, {ci[1]:.6f}]'
        text.append(f"| {row['study']} | {row['policy']} | {point}{interval} | "
            f"{row['supported_record_head_views']}/{row['required_record_head_views']} | "
            f"{row['undefined_easy_record_head_views']} | {row['selected']} ({row['selected_unknown']}) |")
    text += ['', '## Training and Failure Boundaries', '',
        '| Study | Fits | Primary optimizer updates | Seconds | Peak RSS bytes |',
        '|---|---:|---:|---:|---:|']
    for study, fit in value['training'].items():
        text.append(f"| {study} | {fit['fits']} | {fit['updates']} | {fit['seconds']:.6f} | {fit['peak_RSS_bytes']} |")
    text += ['', 'All action-bank arms retain their preregistered seeds and complete roster.',
        'The published deferral loss is an application adapter, not a reproduction of',
        'its original experiments or bounded-loss theorem verification. Its cost-gated',
        'variant is a fixed control, not a calibrated deployment policy.', '',
        'The support-unit repair changes only the numerical unit of the third fitting',
        'constraint; all/easy2% budgets and the support feasible set stay unchanged.',
        'Any success or failure must be interpreted alongside full completion bounds,',
        'every-record risk, the physical-only contrast, and fitting behavior. A zero',
        'intervention count is not a passed selected-risk gate. Published loss values',
        'and primal-dual objectives are different quantities, not cross-method scores.', '',
        'These are frozen-forecaster routing studies, not a new neural trajectory lift',
        'or scene-joint optimizer. Selection, calibration and confirmation remain closed.',
        'M3W remains a2.5D trajectory scaffold; pixel/raw-frame results are not metric',
        'or effective-second claims. No deployment, Stage5C execution or SMC is enabled.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    value = build(); md = render(value)
    dest = PUBLIC/'cost_decision_evidence_20261029'
    outputs = {dest.with_suffix('.json'): json.dumps(value, indent=2)+'\n',
               dest.with_suffix('.md'): md}
    for path, content in outputs.items():
        if args.check:
            if path.read_text() != content:
                raise ValueError('Derived evidence changed')
        elif path.exists():
            if path.read_text() != content:
                raise ValueError('Preserve existing evidence; use a new revision')
        else:
            path.write_text(content)
    print(json.dumps(dict(studies=len(STUDIES), exported_rows=len(value['policies']),
                         exact_export_check=args.check, scientific_recalculation=False)))


if __name__ == '__main__':
    main()
