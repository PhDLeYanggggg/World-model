"""Read completed fixed-grid metrics; describe failures without modifying a policy."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.evaluation import m3w_calibration_failure_taxonomy as api

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'


def checked(ref):
    path = ROOT/ref['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Metric receipt changed')
    return path


def main():
    complete = json.loads((PUBLIC/'eval.json').read_text())
    refs = complete['private_metric_receipts']
    if len(refs) != 918:
        raise ValueError('Complete fixed-grid calibration readout required')
    summary = json.loads(checked(complete['summary']).read_text())
    counts = {}; coverage = {}
    fields = ('rows', 'known_rows', 'wholly_unknown_rows', 'complete12_rows', 'partial_future_rows', 'endpoint_rows')
    for ref in refs:
        row = json.loads(checked(ref).read_text()); member = row['source_member']
        if len(row['scores']) != 52:
            raise ValueError('All13multipliers and4policies required')
        for cell in row['scores']:
            score = cell['metrics']; current = {k: score[k] for k in fields}
            if member in coverage and coverage[member] != current:
                raise ValueError('Policy/producer-dependent label roster is forbidden')
            coverage[member] = current
            if len(score['recordings']) != 1:
                raise ValueError('One original recording per metric receipt required')
            view = next(iter(score['recordings'].values()))
            cause = api.classify(view); key = (cell['multiplier'], cell['arm'])
            item = counts.setdefault(key, dict(exclusive=Counter(), overlapping=Counter()))
            item['exclusive'][cause['category']] += 1
            for k, v in cause.items():
                if k != 'category':
                    item['overlapping'][k] += int(v)
    rows = []
    for curve in summary['curves']:
        for arm, value in curve['policies'].items():
            item = counts[curve['multiplier'], arm]
            if (sum(item['exclusive'].values()) != 918
                    or item['exclusive']['supported'] != value['supported_record_views']):
                raise ValueError('Taxonomy must reconcile to the unchanged original risk gate')
            rows.append(dict(multiplier=curve['multiplier'], arm=arm,
                exclusive={k:item['exclusive'][k] for k in api.CATEGORIES}, overlapping=dict(item['overlapping'])))
    if len(coverage) != 102:
        raise ValueError('Every original recording required')
    full_unsupported = {r['overlapping'].get('full_roster_easy_reference_unavailable', 0) for r in rows}
    if len(full_unsupported) != 1:
        raise ValueError('Full-roster reference availability must not change with action selection')
    out = dict(result_source='fresh_run_descriptive_completed_calibration_metrics_no_policy_change',
        categories=list(api.CATEGORIES), rows=rows, unique_recordings=102,
        unique_label_coverage={k:sum(r[k] for r in coverage.values()) for k in fields},
        repeated_pipeline_record_views=918, empirical_calibration_not_confirmation=True,
        overlapping_failures_are_not_disjoint=True, exclusive_priority=list(api.CATEGORIES),
        chosen_multiplier_unchanged=summary['selection']['selected_multiplier'],
        full_roster_easy_reference_unavailable_views=next(iter(full_unsupported)),
        new_model_or_threshold_fit=False, confirmation_access=False, deployment_changed=False)
    (PUBLIC/'risk_failure_taxonomy.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
    lines = ['# Calibration Risk Failure Taxonomy', '',
        'Descriptive readout only, not another policy or parameter search. All 102 recordings',
        'and all 918 pipeline-record views are retained at every grid point. Exclusive',
        'categories use the listed priority; the JSON also retains overlapping flags.',
        'Known label coverage is reconciled across every policy and producer.', '',
        '|Lambda|Policy|Unsupported|Observed harm|Easy net only|Unknown-bound only|Supported|',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|'+str(r['multiplier'])+'|'+r['arm']+'|'+'|'.join(str(r['exclusive'][k]) for k in api.CATEGORIES)+'|')
    lines += ['', f'Full-roster easy-reference unavailable views: {out["full_roster_easy_reference_unavailable_views"]}/918.',
        'This availability flag is identical across all actions and multipliers.',
        'Where the complete known-easy reference mass is zero, no selected subset',
        'can have positive known-easy reference under the unchanged floor, easy',
        'definition and labels. That is a support obstruction, not something a',
        'different threshold or trajectory predictor can repair on these same labels.',
        '', 'Unsupported reference mass is not a safe zero. A completion-bound-only',
        'failure does not prove missing rows actually incur worst-case harm. Observed',
        'positive harm and signed easy-net degradation are distinct. All figures are',
        'conditional calibration readouts, not independent confirmation or a population',
        'guarantee. Partial-future ADE does not recover unobserved steps.', '']
    (PUBLIC/'risk_failure_taxonomy.md').write_text('\n'.join(lines))
    print(json.dumps(dict(views=918, metric_groups=47736, unique_label_coverage=out['unique_label_coverage'])))


if __name__ == '__main__':
    main()
