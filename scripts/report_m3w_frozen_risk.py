"""Render aggregate frozen-risk evidence; do not rerun models or select policies."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/publication_readiness_2026_09/frozen_risk_forensics_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/frozen_risk_forensics_v1'
STATUSES = ('proven_exceeds', 'known_within', 'indeterminate')


def main():
    path = OUT / 'analysis.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    complete = json.loads((PRIVATE / 'completion.json').read_text())
    if digest != complete['report_sha256']:
        raise ValueError('Completed risk analysis changed')
    report = json.loads(path.read_text())
    rows = report['results']
    if len(rows) != 72 or len({(r['family'], r['candidate'], r['control']) for r in rows}) != 72:
        raise ValueError('All 72 unique fixed comparisons are required')
    if any(r['queries'] != 970 or r['agents'] != 37775 for r in rows):
        raise ValueError('Population changed')

    # A second reduction of receipt-bound per-query records, not the production reducer.
    verified, query_checks = 0, 0
    import numpy as np
    lookup = {(r['family'], r['candidate'], r['control']): r for r in rows}
    for entry in complete['receipts']:
        receipt_path = ROOT / entry['path']
        if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Receipt changed')
        receipt = json.loads(receipt_path.read_text())
        cache_path = receipt_path.with_name(receipt_path.name.replace('.receipt.json', '.json'))
        if hashlib.sha256(cache_path.read_bytes()).hexdigest() != receipt['cache_sha256']:
            raise ValueError('Query details changed')
        if receipt['run_sha256'] != report['run_sha256']:
            raise ValueError('Query details belong to a different run')
        cache = json.loads(cache_path.read_text())
        for control, queries in cache['query_records'].items():
            expected = lookup[(*receipt['key'], control)]
            n = np.array([q['agents'] for q in queries], dtype=np.int64)
            harm = np.array([q['observed_positive_harm_sum'] for q in queries])
            unknown = np.array([q['unknown_selected_agents'] for q in queries], dtype=np.int64)
            budget = np.array([q['budget'] for q in queries])
            lower = harm / n
            statuses = np.where(lower > budget + 1e-10, 'proven_exceeds',
                                np.where(unknown == 0, 'known_within', 'indeterminate'))
            if list(statuses) != [q['budget_status'] for q in queries]:
                raise ValueError('Budget classification replay failed')
            np.testing.assert_allclose(lower.mean(), expected['mean_query_harm_lower_bound'], rtol=1e-12)
            np.testing.assert_allclose(harm.sum()/n.sum(), expected['mean_past_agent_harm_lower_bound'], rtol=1e-12)
            predicted = np.array([q['predicted_mean_harm'] for q in queries])
            if int(np.sum(predicted > budget + 1e-10)) != expected['predicted_budget_failure_queries']:
                raise ValueError('Predicted-budget count differs')
            for status in STATUSES:
                mask = statuses == status
                group = expected['budget_status'][status]
                if int(mask.sum()) != group['queries']:
                    raise ValueError('Status count differs')
                for key in ('easy_positive_harm_sum', 'easy_net_excess_sum', 'easy_baseline_sum'):
                    values = np.array([q[key] for q in queries])
                    np.testing.assert_allclose(values[mask].sum(), group[key], rtol=1e-12, atol=1e-12)
            verified += 1
            query_checks += len(queries)
    if verified != 72:
        raise ValueError('Incomplete second reduction')

    numeric = lambda x: 'unknown' if x is None else f'{x:.8g}'
    short = {'independent_coupling_reference': 'risk-only',
             'unary_geometry_exact_count': 'unary', 'joint_coupling_exact_count': 'joint'}
    lines = ['# All Frozen Risk Comparisons', '',
        'Fresh descriptive reduction of cached, hash-verified frozen decisions. No fitting,',
        'policy selection, independent calibration or new primary metric. Each row reuses',
        'the same 970 queries/37,775 past agents, including 28,324 ADE-labeled agents.',
        'These are repeated comparisons, not independent sample sizes.', '',
        'E = observed lower bound proves the realized budget is exceeded; W = every',
        'selected cost is known and the realized budget is met; ? = indeterminate.',
        'Missing selected costs are never imputed as zero. W is an outcome-defined',
        'diagnostic, not a causal deployment filter. All predicted budgets pass.', '',
        '| Family | Fixed combination | Control | E / W / ? queries | Missing selected costs | Mean predicted harm | Mean observed lower bound | Easy harm from W (%) | Easy degradation (%) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    bin_rows = []
    for r in rows:
        counts = ' / '.join(str(r['budget_status'][s]['queries']) for s in STATUSES)
        fraction = r['easy_positive_harm_fraction']['known_within']
        lines.append(f"| {r['family']} | {r['candidate']} | {short[r['control']]} | {counts} | "
            f"{r['unknown_selected_agents']} | {numeric(r['mean_query_predicted_harm'])} | "
            f"{numeric(r['mean_query_harm_lower_bound'])} | {numeric(None if fraction is None else 100*fraction)} | "
            f"{numeric(r['easy_degradation_percent'])} |")
        for b in r['reliability_bins']:
            bin_rows.append({k: r[k] for k in ('family', 'candidate', 'control')} | b)
    lines.extend(['', 'Means above weight queries equally. The CSV and JSON also retain the distinct',
        'past-agent-weighted lower bound. Costs use the unchanged past-normalized ADE,',
        'not meters, probabilities or the pending native-unit evaluation proposal.',
        'Full bin counts and observed/exact selected-cost means are in `reliability_bins.csv`.',
        'Empty bins remain empty. These are descriptive cost reliability tables, not ECE.', ''])
    (OUT / 'complete_results.md').write_text('\n'.join(lines))
    with (OUT / 'reliability_bins.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bin_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(bin_rows)
    controls_path = OUT / 'all_controls.csv'
    with controls_path.open(newline='') as stream:
        controls = list(csv.reader(stream))
    with controls_path.open('w', newline='') as stream:
        csv.writer(stream, lineterminator='\n').writerows(controls)

    within_excess = sum(r['budget_status']['known_within']['easy_net_excess_sum'] >
        .02*r['budget_status']['known_within']['easy_baseline_sum'] for r in rows)
    summary = dict(analysis_sha256=digest, run_sha256=report['run_sha256'],
        independent_reducer_cells=verified, repeated_query_status_checks=query_checks,
        predicted_budget_failures=sum(r['predicted_budget_failure_queries'] for r in rows),
        cells_with_mean_lower_bound_above_prediction=sum(r['mean_query_harm_lower_bound'] >
            r['mean_query_predicted_harm'] for r in rows),
        cells_with_easy_excess_above_two_percent_inside_known_within_queries=within_excess,
        confirmed_exceedance_query_range=[min(r['budget_status']['proven_exceeds']['queries'] for r in rows),
            max(r['budget_status']['proven_exceeds']['queries'] for r in rows)],
        indeterminate_query_range=[min(r['budget_status']['indeterminate']['queries'] for r in rows),
            max(r['budget_status']['indeterminate']['queries'] for r in rows)],
        missing_selected_cost_range=[min(r['unknown_selected_agents'] for r in rows),
            max(r['unknown_selected_agents'] for r in rows)],
        known_within_easy_harm_share_percent_range=[
            100*min(r['easy_positive_harm_fraction']['known_within'] for r in rows),
            100*max(r['easy_positive_harm_fraction']['known_within'] for r in rows)],
        new_training=False, new_inference=False, primary_metric_changed=False,
        independent_calibration=False, policy_selection=False)
    (OUT / 'verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    os.environ.setdefault('MPLCONFIGDIR', str(PRIVATE / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'm3w-risk-v1'})
    joint = [r for r in rows if r['control'] == 'joint_coupling_exact_count']
    labels = [r['family'].replace('transformer','TF').replace('eqmotion','Eq')+' '+
              r['candidate'].replace('seed','s').replace('_',' / ') for r in joint]
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    colors = ('#b53b44', '#327d61', '#8d939c')
    names = ('Proven budget exceedance', 'Known within budget', 'Indeterminate')
    y = np.arange(len(joint))
    for ax, field in zip(axes, ('queries', 'easy_harm')):
        left = np.zeros(len(joint))
        for status, color, name in zip(STATUSES, colors, names):
            values = np.array([100*r['budget_status'][status]['queries']/r['queries'] if field == 'queries'
                else 100*r['easy_positive_harm_fraction'][status] for r in joint])
            ax.barh(y, values, left=left, color=color, height=.78, label=name)
            left += values
        ax.set_xlim(0, 100)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        ax.set_xlabel('Share (%)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title('Realized query-budget status')
    axes[1].set_title('Where observed easy positive harm occurs')
    fig.suptitle('Frozen joint controls: predicted budgets pass, easy preservation fails', fontsize=14)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, bbox_to_anchor=(.6, .045), frameon=False)
    fig.text(.5, .018, '24 fixed combinations; 970 repeated queries each; one explored physical site.\n'
             'Missing selected labels remain unknown. No policy selection or independent calibration.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.105,1,.96))
    fig.savefig(OUT / 'risk_attribution.svg', metadata={'Date': None})
    svg = OUT / 'risk_attribution.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(PRIVATE / 'risk_attribution_preview.png', dpi=130)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
