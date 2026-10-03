"""Describe verified frozen support findings without selecting a model or policy."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_m3w_cost_support_diagnostic import PUBLIC, PARENT, sha


def average(values):
    return math.fsum(values)/len(values) if values else None


def summarize():
    receipt = json.loads((PUBLIC/'verification.json').read_text())
    assert receipt['summary_sha256'] == sha(PUBLIC/'summary.json')
    assert receipt['complete_sha256'] == sha(PUBLIC/'complete.json')
    complete = json.loads((PUBLIC/'complete.json').read_text())
    parent = json.loads((PARENT/'complete.json').read_text())
    anchors = {}
    for ref in parent['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text())
        anchors[r['group'], r['head_seed']] = r
    rows = []
    for ref in complete['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    assert len(rows) == 72
    result = dict(result_source='fresh_aggregate_of_cached_verified_diagnostics', groups=72,
        verification_sha256=sha(PUBLIC/'verification.json'), new_training=False,
        policy_selection=False, independent_confirmation=False, deployment_changed=False)
    result['head_sign_counts'] = {
        'train_tree_decreased': sum(r['tree_scores']['train']['mean_tree_MSE_change'] < 0 for r in rows),
        'train_raw_ensemble_decreased': sum(r['tree_scores']['train']['raw_ensemble_MSE_change'] < 0 for r in rows),
        'train_projected_ensemble_decreased': sum(r['training_projected_error_change']['global_weighted_MSE_change'] < 0 for r in rows),
        'validation_tree_increased': sum(r['tree_scores']['validation']['mean_tree_MSE_change'] > 0 for r in rows),
        'validation_raw_ensemble_increased': sum(r['tree_scores']['validation']['raw_ensemble_MSE_change'] > 0 for r in rows),
        'validation_projected_ensemble_increased': sum(r['diagnosis']['cohorts']['all']['global_weighted_MSE_change'] > 0 for r in rows)}
    result['localities'] = {}
    for source in sorted({r['source'] for r in rows}):
        rr = [r for r in rows if r['source'] == source]
        d = dict(heads=len(rr))
        for role in ('train', 'validation'):
            for key in ('mean_tree_MSE_change', 'raw_ensemble_MSE_change', 'dispersion_change'):
                d[role+'_'+key] = average([r['tree_scores'][role][key] for r in rr])
        d['train_projected_MSE_change'] = average([r['training_projected_error_change']['global_weighted_MSE_change'] for r in rr])
        d['validation_projected_MSE_change'] = average([r['diagnosis']['cohorts']['all']['global_weighted_MSE_change'] for r in rr])
        for key in ('outside_majority_leaf_quality_box', 'inside_majority_leaf_quality_box',
                    'harm_all_train_zero', 'easy_harm_all_train_zero'):
            d[key] = dict(rows=sum(r['diagnosis']['strata'][key]['rows'] for r in rr),
                MSE_change=average([r['diagnosis']['strata'][key]['global_weighted_MSE_change'] for r in rr]))
        result['localities'][source] = d
    failures = []
    for r in rows:
        a = anchors[r['group'], r['head_seed']]
        policy = a['result']['policies']['cost']
        upper = policy['easy_selected_risk_upper']
        if upper is None or upper <= .02+1e-12:
            continue
        failures.append(dict(group=r['group'], source=r['source'], head_seed=r['head_seed'],
            selected_count=policy['selected_count'], selected_unknown=policy['selected_unknown'],
            easy_selected_risk_upper=upper,
            actions_equal_original=a['action_hashes']['cost'] == a['action_hashes']['original'],
            support_cohorts={k: r['diagnosis']['support_cohorts'][k] for k in
                ('cost_selected', 'selected_known_harm', 'selected_known_easy_harm', 'selected_unknown')}))
    result['upper_failure_support'] = failures
    return result


def table(result):
    lines = ['# Frozen Support and Objective Findings', '',
        'Post-hoc development diagnosis of frozen predictions, not new training, policy selection,',
        'independent confirmation or a causal ablation. Counts across heads are repeated occurrences.', '',
        '## Optimization Versus Generalization', '', '| Quantity | Heads / 72 |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in result['head_sign_counts'].items()]
    lines += ['', '| Locality | TRAIN raw ensemble change | TRAIN projected change | Validation raw change | Validation projected change |',
              '|---|---:|---:|---:|---:|']
    for site, d in result['localities'].items():
        vals = [d[k] for k in ('train_raw_ensemble_MSE_change', 'train_projected_MSE_change',
                               'validation_raw_ensemble_MSE_change', 'validation_projected_MSE_change')]
        lines.append('| '+site+' | '+' | '.join(f'{v:+.6f}' for v in vals)+' |')
    lines += ['', 'Negative change is better normalized signed-score MSE. These are not ADE/FDE gains.',
        'Do not add overlapping support strata as separate causes. Leaf quality boxes are TRAIN',
        'min/max descriptors, not guarantees, prediction intervals or new action filters.', '',
        '## Upper-Risk Failure Support', '',
        '| Source / seed / producer | Selected | Unknown | Upper % | Original actions retained | Selected known EH rows | Mean zero-TRAIN-EH tree fraction | Mean outside-quality tree fraction |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in result['upper_failure_support']:
        c = r['support_cohorts']['selected_known_easy_harm']
        means = c['descriptor_mean']
        val = lambda k: 'undefined' if means[k] is None else f'{means[k]:.6f}'
        lines.append(f"| {r['group']} / {r['head_seed']} | {r['selected_count']} | {r['selected_unknown']} | "
            f"{100*r['easy_selected_risk_upper']:.4f} | {r['actions_equal_original']} | {c['rows']} | "
            f"{val('zero_train_easy_harm_fraction')} | {val('outside_train_quality_box_fraction')} |")
    lines += ['', 'Upper risk includes completion for unknown outcomes and is not observed harm.',
        'Unavailable future labels are retained as unknown in policy evaluation; label-based',
        'cohorts above are offline diagnosis only. Stage5C/SMC remain off.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    result = summarize()
    for name, payload in [('findings.json', json.dumps(result, indent=2)+'\n'), ('findings.md', table(result))]:
        path = PUBLIC/name
        if path.exists():
            assert path.read_text() == payload
        else:
            path.write_text(payload)
    print(json.dumps(result['head_sign_counts']))
