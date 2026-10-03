"""Post-run tables for the frozen one-factor extension control; no selection."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_m3w_leaf_quality_extension import PUBLIC, sha, ci


def mean(values):
    return math.fsum(values)/len(values)


def summarize():
    receipt = json.loads((PUBLIC/'verification.json').read_text())
    for key in ('summary', 'complete'):
        assert sha(PUBLIC/(key+'.json')) == receipt[key+'_sha256']
    rows = []
    for ref in json.loads((PUBLIC/'complete.json').read_text())['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    cfg = json.loads((ROOT/'configs/m3w_european_leaf_quality_extension_v1.json').read_text())
    def interval(get):
        rr = [dict(source=r['source'], result=dict(contrasts=dict(value=get(r)))) for r in rows]
        return ci(rr, 'value', cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    out = dict(groups=len(rows), result_source='fresh_aggregate_cached_verified_extension',
        verification_sha256=sha(PUBLIC/'verification.json'), new_training=False,
        independent_confirmation=False, deployment_changed=False,
        raw_extended_minus_cost_MSE=interval(lambda r: r['raw_scores']['extended']-r['raw_scores']['cost']),
        raw_extended_minus_original_MSE=interval(lambda r: r['raw_scores']['extended']-r['raw_scores']['original']),
        changed_query_fraction=interval(lambda r: r['validation_changed_rows']/r['validation_rows']),
        global_MSE_change_by_cohort={k: interval(lambda r: r['slices'][k]['global_weighted_MSE_change'])
                                     for k in rows[0]['slices']},
        extended_prediction_changed_heads=sum(r['prediction_hashes']['cost'] != r['prediction_hashes']['extended'] for r in rows),
        extended_action_changed_heads=sum(r['action_hashes']['cost'] != r['action_hashes']['extended'] for r in rows),
        training_identity_heads=sum(r['known_training_prediction_hashes']['cost'] == r['known_training_prediction_hashes']['extended'] for r in rows),
        validation_MSE_improved_vs_cost_heads=sum(r['result']['contrasts']['extended_minus_cost_signed_MSE'] < 0 for r in rows),
        validation_MSE_improved_vs_original_heads=sum(r['result']['contrasts']['extended_minus_original_signed_MSE'] < 0 for r in rows))
    out['localities'] = {}
    for site in sorted({r['source'] for r in rows}):
        rr = [r for r in rows if r['source'] == site]
        d = dict(heads=len(rr), quality_changed_fraction=mean([r['validation_changed_rows']/r['validation_rows'] for r in rr]),
            raw_MSE_extended_minus_cost=mean([r['raw_scores']['extended']-r['raw_scores']['cost'] for r in rr]))
        for other in ('original', 'cost', 'additive', 'poisson'):
            for suffix in ('signed_MSE', 'full_utility_percent', 'matched_utility_percent'):
                key = 'extended_minus_'+other+'_'+suffix
                d[key] = mean([r['result']['contrasts'][key] for r in rr])
        out['localities'][site] = d
    out['upper_failure_heads'] = []
    for r in rows:
        a = r['result']['policies']['extended']
        if a['easy_selected_risk_upper'] is not None and a['easy_selected_risk_upper'] > .02+1e-12:
            out['upper_failure_heads'].append(dict(group=r['group'], source=r['source'], head_seed=r['head_seed'],
                selected=a['selected_count'], unknown_selected=a['selected_unknown'],
                known_risk=(a['selected_known_easy_harm_mass']/a['selected_known_easy_reference_mass']
                            if a['selected_known_easy_reference_mass'] > 0 else None),
                upper_risk=a['easy_selected_risk_upper'],
                actions_equal_cost=r['action_hashes']['extended'] == r['action_hashes']['cost'],
                actions_equal_original=r['action_hashes']['extended'] == r['action_hashes']['original']))
    return out


def table(result):
    summary = json.loads((PUBLIC/'summary.json').read_text())
    lines = ['# Leaf-Quality Extension Results', '',
        'One preregistered extension, no refit or threshold search. The frozen controls are',
        'cached_verified; all extension inference and policy readouts are fresh_run.',
        'This is exposed development evidence, not independent confirmation.', '',
        '| Arm | Selected | Unknown | Complete support / 72 | Known violations | Upper violations | Worst upper % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k in ('original', 'additive', 'poisson', 'cost', 'extended'):
        r = summary[k]
        upper = 'undefined' if r['worst_easy_upper'] is None else f"{100*r['worst_easy_upper']:.4f}"
        lines.append(f"| {k} | {r['selected']} | {r['unknown_selected']} | {r['complete_support']} | "
                     f"{r['known_label_violations']} | {r['violations']} | {upper} |")
    lines += ['', '| Extension contrast | Mean | Nominal 95% locality CI |', '|---|---:|---|']
    for k, r in summary.items():
        if k.startswith('extended_minus_'):
            interval = 'undefined' if r['CI95'] is None else f"[{r['CI95'][0]:+.8f}, {r['CI95'][1]:+.8f}]"
            value = 'undefined' if r['mean'] is None else f"{r['mean']:+.8f}"
            lines.append(f'| {k} | {value} | {interval} |')
    lines += ['', 'Utility is a percentage of full known reference error mass, not ADE/FDE gain.',
        'The 3,000-resample intervals use locality blocks; repeated heads and windows are not',
        'independent units. Intervals are nominal, not adjusted for prior development search.', '',
        '| Locality | Quality changed fraction | MSE vs cost | MSE vs original | Full utility vs original % |',
        '|---|---:|---:|---:|---:|']
    for site, r in result['localities'].items():
        lines.append(f"| {site} | {r['quality_changed_fraction']:.6f} | {r['extended_minus_cost_signed_MSE']:+.6f} | "
            f"{r['extended_minus_original_signed_MSE']:+.6f} | {r['extended_minus_original_full_utility_percent']:+.6f} |")
    lines += ['', '## Registered Screen', '', f"Advance to transfer: {summary['advance_to_transfer']}.",
        f"Every head supported and within the absolute easy-risk budget: {summary['all_heads_easy_supported_within_budget']}.", '',
        *['- '+v for v in summary['failure_reasons']], '',
        'No deployment change, Stage5C or SMC. Image-local rawstride12 obs8/pred12, detector-silver.',
        'No seconds, metric, physical safety, true 3D, foundation or submission-ready claim.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    result = summarize()
    for name, text in [('findings.json', json.dumps(result, indent=2)+'\n'), ('findings.md', table(result))]:
        path = PUBLIC/name
        if path.exists():
            assert path.read_text() == text
        else:
            path.write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k.endswith('_heads')}, indent=2))
