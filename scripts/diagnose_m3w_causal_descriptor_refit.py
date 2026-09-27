"""Post-readout paired diagnostics; never changes models, actions or gates."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def paired_quality(rows):
    index = {}
    for row in rows:
        key = (row['group'], row['site'], row['policy'])
        if key in index:
            raise ValueError('Duplicate quality view')
        index[key] = row
    new = {k[:2] for k in index if k[2] == 'descriptor'}
    control = {k[:2] for k in index if k[2] == 'control'}
    if new != control or not new:
        raise ValueError('Paired source roster mismatch')
    result = []
    for group, site in sorted(new):
        a = index[(group, site, 'descriptor')]
        b = index[(group, site, 'control')]
        if a['seed'] != b['seed'] or set(a['metric']) != set(b['metric']):
            raise ValueError('Mismatched quality fields or seeds')
        values = {k: None if a['metric'][k] is None or b['metric'][k] is None
                  else a['metric'][k] - b['metric'][k] for k in a['metric']}
        for part in ('all', 'easy'):
            held, fit = values[part + '_signed_MSE'], values[part + '_fit_MSE']
            values[part + '_generalization_gap_difference'] = (
                None if held is None or fit is None else held - fit)
        result.append(dict(group=group, site=site, seed=a['seed'], metric=values))
    return result


def main():
    from scripts import run_m3w_european_causal_descriptor_refit as run
    from scripts.report_m3w_fixed_floor_tail import ci
    cfg = json.loads((ROOT/run.CONFIG).read_text())
    receipt = json.loads((run.PUBLIC/'evaluation_replay.json').read_text())
    assert receipt['exact']
    for key in ('summary', 'details'):
        assert run.base.artifact(ROOT/receipt[key]['path']) == receipt[key]
    details = json.loads((run.PRIVATE/'details.json').read_text())
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    paired = paired_quality(details['quality'])
    sites = sorted({r['site'] for r in paired})
    estimates = {k: run.base.inter.paired_localities(paired, sites, k,
        cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for k in paired[0]['metric']}
    risks = {}
    for policy in ('control', 'descriptor', 'control_matched_count'):
        rows = [r for r in details['rows'] if r['policy'] == policy]
        defined = [r for r in rows if r['metric']['selected_positive_harm_ratio'] is not None]
        worst = max(defined, key=lambda r: r['metric']['selected_positive_harm_ratio'])
        risks[policy] = dict(
            undefined_views_by_site={s: sum(r['site'] == s and r['metric']['selected_positive_harm_ratio'] is None
                for r in rows) for s in sites},
            violating_views_by_site={s: sum(r['site'] == s and r['metric']['selected_positive_harm_ratio'] is not None
                and r['metric']['selected_positive_harm_ratio'] > .02 for r in rows) for s in sites},
            worst_defined_view=worst,
            worst_known_selected_rows=int(round(worst['metric']['known_intervention_rate'] * worst['metric']['known_rows'])))
    equal = summary['paired']['control_matched_count']['ADE_gain_percent']['by_site']
    payload = dict(result_source='fresh_post_readout_diagnostic_of_verified_aggregates',
        input_artifacts=receipt, sign='new minus control; lower MSE is better',
        quality_difference=estimates, risks=risks, equal_count_ADE_by_site=equal,
        positive_equal_count_localities=sum(v is not None and v > 0 for v in equal.values()),
        negative_equal_count_localities=sum(v is not None and v < 0 for v in equal.values()),
        primary_replaced=False, model_selection=False, action_changes=False,
        independent_confirmation=False, calibration_certificate=False)
    run.base.immutable_json(run.PUBLIC/'post_readout_diagnosis.json', payload)
    lines = ['# Causal-Descriptor Failure Diagnosis', '',
        'Post-readout, descriptive analysis of hash-verified aggregates. This does not replace the registered primary, choose a model, or change an action.', '',
        '## Paired Score Error', '',
        'New minus signed-excess control; negative is better. Same 3,000 locality-bootstrap draws, averaging dependent views within each of 12 opened localities.', '',
        '| Quantity | Paired difference [95% exploratory CI] |', '|---|---:|']
    for key, value in estimates.items():
        lines.append('| ' + key + ' | ' + ci(value) + ' |')
    lines += ['', 'Training-fit improvement is not held-source calibration. The four output components are not separately identified cost moments.', '',
        '## Same-Query Selection Quality', '',
        '| Held development locality | ADE advantage over count-matched control (%) |', '|---|---:|']
    for site, value in equal.items():
        lines.append(f'| {site} | {value:.6f} |' if value is not None else f'| {site} | undefined |')
    lines += ['', 'Each query fixes the same intervention count, eligible pool and row-ID tie break. This is an ordering diagnostic, not a new deployable rule.', '',
        '## Risk Failures', '',
        '| Policy | Worst defined harm (%) | Known selected rows in that view | Empty views |', '|---|---:|---:|---:|']
    for policy, value in risks.items():
        lines.append(f"| {policy} | {100*value['worst_defined_view']['metric']['selected_positive_harm_ratio']:.6f} | {value['worst_known_selected_rows']} | {sum(value['undefined_views_by_site'].values())} |")
    lines += ['', 'View counts are dependent source/producer/fit/seed views, not independent scenes. Empty denominators stay undefined. Unknown-label actions cannot be claimed safe.', '',
        '## Interpretation', '',
        'The added descriptors change intervention coverage and slightly improve average error, but do not improve equal-budget ranking. Easy net preservation coexists with positive-harm budget violations. This falsifies the simple descriptor-only repair in this setup, not the usefulness of all causal motion features.',
        'The evidence localizes a decision-quality and conditional-risk problem; it does not establish whether loss identifiability, imperfect utility scores, label noise or source shift is the sole cause. The 384 additional parameters also prevent an equal-capacity semantic claim.',
        'Next: freeze these actions and decompose benefit versus positive harm in the selections that differ from the same-query control, including unknown-label and sparse-view support. Then register one targeted gain/harm ordering repair. Do not loosen the budget or pick a favorable held slice.', '',
        'Image-local detector silver, obs8/pred12 rawstride12. No independent confirmation, deployment, metric, seconds, physical-safety, true3D or foundation claim. Stage5C and SMC remain disabled.']
    (run.PUBLIC/'failure_analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(paired_views=len(paired), quality_difference=estimates,
        positive_equal_count_localities=payload['positive_equal_count_localities'])))


if __name__ == '__main__':
    main()
